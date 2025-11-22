"""
Base Trading Strategy Module
Contains BaseStrategy class with shared functionality for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal
import pandas as pd

from trading.data import fetch_binance_data
from trading.indicators import calculate_indicators
from trading.patterns import detect_candlestick_patterns, get_candlestick_signals
from trading.analysis import (
    get_multi_indicator_confirmation,
    get_volume_confirmation,
    find_confluence_zones,
    get_dynamic_thresholds,
    check_divergence,
    detect_market_regime,
    analyze_historical_performance,
)
from trading.risk import (
    calculate_risk_score,
    calculate_volatility_adjusted_risk,
    calculate_support_resistance,
    calculate_fibonacci_levels,
)


class BaseStrategy(ABC):
    """
    Abstract Base Class for Trading Strategies

    Provides common functionality for:
    - Data fetching from Binance
    - Technical indicator calculations
    - Market analysis
    - Risk management
    - Support/Resistance and Fibonacci calculations

    Subclasses must implement:
    - get_signal(): Generate trading signals
    - get_recommendation(): Display trading recommendation
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        leverage: int = 5,
        timeframes: Optional[dict[str, str]] = None
    ):
        """
        Initialize base strategy

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            leverage: Trading leverage multiplier
            timeframes: Dict of timeframe names to Binance intervals
        """
        self.symbol = symbol
        self.leverage = leverage
        self.timeframes = timeframes or {}
        self.data: dict[str, pd.DataFrame] = {}
        self._last_analysis: Optional[dict] = None

    # === DATA FETCHING ===

    def fetch_data(self, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Binance

        Args:
            timeframe: Binance interval (e.g., "1d", "4h", "1w")
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data or None if error
        """
        return fetch_binance_data(self.symbol, timeframe, limit)

    # === INDICATOR CALCULATIONS ===

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with calculated indicators
        """
        return calculate_indicators(df)

    # === PATTERN DETECTION ===

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns in data"""
        return detect_candlestick_patterns(df)

    def get_candlestick_signals(self, df: pd.DataFrame) -> dict:
        """Get trading signals from candlestick patterns"""
        return get_candlestick_signals(df)

    # === MARKET ANALYSIS ===

    def get_multi_indicator_confirmation(self, df: pd.DataFrame) -> dict:
        """Get multi-indicator confirmation analysis"""
        return get_multi_indicator_confirmation(df)

    def get_volume_confirmation(self, df: pd.DataFrame) -> dict:
        """Get volume-based confirmation analysis"""
        return get_volume_confirmation(df)

    def find_confluence_zones(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> dict:
        """Find price confluence zones"""
        return find_confluence_zones(df, current_price)

    def get_dynamic_thresholds(self, df: pd.DataFrame) -> dict:
        """Get volatility-adjusted indicator thresholds"""
        return get_dynamic_thresholds(df)

    def check_divergence(
        self,
        df: pd.DataFrame,
        indicator: str = "RSI",
        lookback: int = 14
    ) -> tuple[Optional[str], float]:
        """
        Check for price-indicator divergence

        Args:
            df: DataFrame with price and indicator data
            indicator: Indicator to check ("RSI", "MACD", etc.)
            lookback: Number of periods to analyze

        Returns:
            Tuple of (divergence_type, strength)
            divergence_type: "bullish", "bearish", or None
            strength: 0-100 indicating divergence strength
        """
        return check_divergence(df, indicator, lookback)

    def detect_market_regime(self, df: pd.DataFrame) -> dict:
        """
        Detect current market regime

        Returns dict with:
            - regime: STRONG_UPTREND, STRONG_DOWNTREND, CONSOLIDATION, etc.
            - confidence: 0-100
            - adx, bb_width, atr_percent, price_range_pct
        """
        return detect_market_regime(df)

    def analyze_historical_performance(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> dict:
        """
        Analyze historical signal performance

        Returns dict with:
            - win_rate: Historical win percentage
            - avg_return: Average return per signal
            - max_drawdown: Maximum drawdown
            - sharpe: Sharpe ratio
        """
        return analyze_historical_performance(df, lookback)

    # === RISK MANAGEMENT ===

    def calculate_risk_score(
        self,
        df: pd.DataFrame,
        signal_type: Literal["LONG", "SHORT", "NEUTRAL"]
    ) -> dict:
        """
        Calculate trade risk score

        Returns dict with:
            - score: 0-100 (lower is better)
            - level: LOW, MEDIUM, HIGH
            - factors: List of risk factors
        """
        return calculate_risk_score(df, signal_type)

    def calculate_volatility_adjusted_risk(
        self,
        df: pd.DataFrame,
        base_risk_pct: float = 2.0
    ) -> dict:
        """
        Calculate volatility-adjusted risk percentage

        Returns dict with:
            - adjusted_risk_pct: Risk % adjusted for volatility
            - volatility_ratio: Current vs average volatility
            - risk_note: Description of adjustment
        """
        return calculate_volatility_adjusted_risk(df, base_risk_pct)

    def calculate_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> dict:
        """
        Calculate support and resistance levels

        Returns dict with:
            - resistance: List of resistance levels
            - support: List of support levels
            - main_resistance: Strongest resistance
            - main_support: Strongest support
        """
        return calculate_support_resistance(df, lookback)

    def calculate_fibonacci_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> tuple[dict, str]:
        """
        Calculate Fibonacci retracement levels

        Returns:
            Tuple of (fib_levels_dict, trend_direction)
        """
        return calculate_fibonacci_levels(df, lookback)

    # === TREND ANALYSIS ===

    def get_trend_strength(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Calculate trend strength score

        Args:
            df: DataFrame with indicators

        Returns:
            Tuple of (score, max_score)
            Positive = bullish, Negative = bearish
        """
        latest = df.iloc[-1]
        score = 0
        max_score = 10

        # EMA alignment
        if latest["EMA_9"] > latest["EMA_21"] > latest["EMA_50"]:
            score += 2
        elif latest["EMA_9"] < latest["EMA_21"] < latest["EMA_50"]:
            score -= 2

        # Price vs EMAs
        if latest["close"] > latest["EMA_9"] > latest["EMA_21"]:
            score += 1
        elif latest["close"] < latest["EMA_9"] < latest["EMA_21"]:
            score -= 1

        # ADX trend strength
        if latest["ADX"] > 25:
            if latest["DI_plus"] > latest["DI_minus"]:
                score += 2
            else:
                score -= 2
        elif latest["ADX"] < 20:
            score = int(score * 0.5)

        # Supertrend
        if pd.notna(latest.get("SUPERTREND_DIR")):
            if latest["SUPERTREND_DIR"] == 1:
                score += 1
            else:
                score -= 1

        # MACD
        if latest["MACD"] > latest["MACD_signal"] and latest["MACD_histogram"] > 0:
            score += 1
        elif latest["MACD"] < latest["MACD_signal"] and latest["MACD_histogram"] < 0:
            score -= 1

        return score, max_score

    def check_trend_consistency(self, lookback: int = 10) -> dict:
        """
        Check trend consistency across timeframes

        Returns dict with:
            - consistent: bool
            - direction: "bullish", "bearish", "neutral", "mixed"
            - score: 0-100
        """
        if not self.data:
            return {"consistent": False, "direction": "neutral", "score": 0}

        scores = {"bullish": 0, "bearish": 0}

        # Weights by timeframe importance
        weights = self._get_timeframe_weights()

        for tf, weight in weights.items():
            if tf not in self.data or self.data[tf] is None:
                continue

            df = self.data[tf]
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)
            latest = df.iloc[-1]

            # EMA Trend
            ema_short = latest.get("EMA_9", latest.get("EMA_12"))
            ema_long = latest.get("EMA_21", latest.get("EMA_26"))

            if ema_short is not None and ema_long is not None:
                if ema_short > ema_long:
                    scores["bullish"] += weight * 2
                else:
                    scores["bearish"] += weight * 2

            # MACD
            if latest["MACD"] > latest["MACD_signal"]:
                scores["bullish"] += weight
            else:
                scores["bearish"] += weight

            # Price vs EMA
            if ema_long is not None:
                if latest["close"] > ema_long:
                    scores["bullish"] += weight
                else:
                    scores["bearish"] += weight

            # Higher Highs / Lower Lows
            higher_highs = sum(
                1 for i in range(1, len(recent))
                if recent.iloc[i]["high"] > recent.iloc[i-1]["high"]
            )
            lower_lows = sum(
                1 for i in range(1, len(recent))
                if recent.iloc[i]["low"] < recent.iloc[i-1]["low"]
            )

            if higher_highs > len(recent) * 0.6:
                scores["bullish"] += weight
            if lower_lows > len(recent) * 0.6:
                scores["bearish"] += weight

        total_score = scores["bullish"] + scores["bearish"]
        if total_score == 0:
            return {"consistent": False, "direction": "neutral", "score": 0}

        bullish_pct = scores["bullish"] / total_score * 100
        bearish_pct = scores["bearish"] / total_score * 100

        if bullish_pct >= 70:
            return {"consistent": True, "direction": "bullish", "score": bullish_pct}
        elif bearish_pct >= 70:
            return {"consistent": True, "direction": "bearish", "score": bearish_pct}
        else:
            return {
                "consistent": False,
                "direction": "mixed",
                "score": max(bullish_pct, bearish_pct)
            }

    # === CONFIDENCE CALCULATION ===

    def get_confidence_level(
        self,
        signals: dict[str, int]
    ) -> tuple[str, float]:
        """
        Calculate confidence level from signals

        Args:
            signals: Dict with "long", "short", "neutral" scores

        Returns:
            Tuple of (recommendation, confidence_pct)
        """
        total = signals["long"] + signals["short"] + signals["neutral"]
        if total == 0:
            return "WAIT", 0

        long_pct = signals["long"] / total * 100
        short_pct = signals["short"] / total * 100

        if long_pct >= 70:
            return "STRONG_LONG", long_pct
        elif long_pct >= 55:
            return "LONG", long_pct
        elif short_pct >= 70:
            return "STRONG_SHORT", short_pct
        elif short_pct >= 55:
            return "SHORT", short_pct
        else:
            return "WAIT", max(long_pct, short_pct)

    # === ABSTRACT METHODS ===

    @abstractmethod
    def _get_timeframe_weights(self) -> dict[str, int]:
        """
        Get weights for each timeframe (must be implemented by subclass)

        Returns:
            Dict mapping timeframe names to weight values
        """
        pass

    @abstractmethod
    def analyze_multi_timeframe(self) -> Optional[bool]:
        """
        Fetch and analyze data across multiple timeframes

        Returns:
            True if successful, None if failed
        """
        pass

    @abstractmethod
    def get_signal(self) -> tuple[dict, dict]:
        """
        Generate trading signals

        Returns:
            Tuple of (signals_dict, reasons_dict)
        """
        pass

    @abstractmethod
    def get_recommendation(self, balance: float) -> None:
        """
        Display trading recommendation

        Args:
            balance: Account balance for position sizing
        """
        pass

    # === UTILITY METHODS ===

    def _format_price(self, price: float) -> str:
        """Format price with appropriate precision"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:.8f}"

    def _calculate_position_size(
        self,
        balance: float,
        risk_pct: float,
        stop_loss_pct: float
    ) -> dict:
        """
        Calculate position size based on risk

        Args:
            balance: Account balance
            risk_pct: Risk percentage per trade
            stop_loss_pct: Stop loss distance as percentage

        Returns:
            Dict with margin_required, position_size
        """
        risk_amount = balance * (risk_pct / 100)
        position_size = (risk_amount / (abs(stop_loss_pct) / 100)) * self.leverage
        margin_required = position_size / self.leverage

        return {
            "risk_amount": risk_amount,
            "position_size": position_size,
            "margin_required": margin_required,
        }
