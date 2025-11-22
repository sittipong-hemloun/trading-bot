"""
Trading Strategies Module
Re-exports strategy classes from the trading package

Usage:
    from strategies import WeeklyTradingStrategy, MonthlyTradingStrategy

All logic has been moved to the trading/ package:
- trading/weekly_strategy.py: WeeklyTradingStrategy class
- trading/monthly_strategy.py: MonthlyTradingStrategy class
- trading/indicators.py: Technical indicator calculations
- trading/patterns.py: Candlestick pattern detection
- trading/analysis.py: Market analysis functions
- trading/risk.py: Risk management functions
- trading/data.py: Data fetching functions
"""

from trading import WeeklyTradingStrategy, MonthlyTradingStrategy

__all__ = ["WeeklyTradingStrategy", "MonthlyTradingStrategy"]
