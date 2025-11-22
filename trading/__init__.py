"""
Trading Module
Complete trading strategy package with all components

Usage:
    from trading import WeeklyTradingStrategy, MonthlyTradingStrategy

    # Or import specific utilities
    from trading.indicators import calculate_indicators
    from trading.analysis import detect_market_regime

    # Or use the base class for custom strategies
    from trading import BaseStrategy
"""

# Base strategy class
from trading.base_strategy import BaseStrategy

# Main strategy classes
from trading.weekly_strategy import WeeklyTradingStrategy
from trading.monthly_strategy import MonthlyTradingStrategy

# Data fetching
from trading.data import fetch_binance_data

# Technical indicators
from trading.indicators import calculate_indicators

# Candlestick patterns
from trading.patterns import detect_candlestick_patterns, get_candlestick_signals

# Market analysis
from trading.analysis import (
    get_multi_indicator_confirmation,
    get_volume_confirmation,
    find_confluence_zones,
    get_dynamic_thresholds,
    check_divergence,
    detect_market_regime,
    analyze_historical_performance,
)

# Risk management
from trading.risk import (
    calculate_risk_score,
    calculate_volatility_adjusted_risk,
    calculate_support_resistance,
    calculate_fibonacci_levels,
)

__all__ = [
    # Base class
    "BaseStrategy",
    # Strategy classes
    "WeeklyTradingStrategy",
    "MonthlyTradingStrategy",
    # Data
    "fetch_binance_data",
    # Indicators
    "calculate_indicators",
    # Patterns
    "detect_candlestick_patterns",
    "get_candlestick_signals",
    # Analysis
    "get_multi_indicator_confirmation",
    "get_volume_confirmation",
    "find_confluence_zones",
    "get_dynamic_thresholds",
    "check_divergence",
    "detect_market_regime",
    "analyze_historical_performance",
    # Risk
    "calculate_risk_score",
    "calculate_volatility_adjusted_risk",
    "calculate_support_resistance",
    "calculate_fibonacci_levels",
]
