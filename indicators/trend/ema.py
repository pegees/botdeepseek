"""EMA (Exponential Moving Average) crossover indicator with slope analysis."""
from typing import Dict, List, Any, Optional
import math
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class EMAIndicator(IndicatorBase):
    """
    EMA crossover indicator with slope/angle analysis.

    Uses fast (9) and slow (21) EMAs to detect trend direction
    and momentum through crossovers and slope analysis.

    Signals:
    - BULLISH: Fast EMA crosses above slow EMA, or strong upward slope
    - BEARISH: Fast EMA crosses below slow EMA, or strong downward slope
    - NEUTRAL: EMAs converging or no clear trend
    """

    @property
    def name(self) -> str:
        return "ema"

    def default_params(self) -> Dict[str, Any]:
        return {
            "fast_period": 9,
            "slow_period": 21,
            "slope_lookback": 3,  # Candles to measure slope
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Calculate EMA crossover and slope.

        Args:
            candles: List of OHLCV dicts with 'close' key
            params: Optional override for fast_period, slow_period, slope_lookback

        Returns:
            IndicatorResult with EMA data and signal
        """
        p = {**self.default_params(), **(params or {})}
        fast_period = p["fast_period"]
        slow_period = p["slow_period"]
        slope_lookback = p["slope_lookback"]

        min_candles = slow_period + slope_lookback
        if len(candles) < min_candles:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        closes = [c["close"] for c in candles]

        # Calculate EMAs
        fast_ema = self._calculate_ema(closes, fast_period)
        slow_ema = self._calculate_ema(closes, slow_period)

        if len(fast_ema) < 2 or len(slow_ema) < 2:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "EMA calculation failed"}
            )

        # Align arrays
        offset = len(fast_ema) - len(slow_ema)

        current_fast = fast_ema[-1]
        current_slow = slow_ema[-1]
        prev_fast = fast_ema[-2]
        prev_slow = slow_ema[-2]

        # Current price for reference
        current_price = closes[-1]

        # Detect crossover
        crossover = None
        if prev_fast <= prev_slow and current_fast > current_slow:
            crossover = "bullish"
        elif prev_fast >= prev_slow and current_fast < current_slow:
            crossover = "bearish"

        # Calculate slope (angle) of fast EMA
        slope = self._calculate_slope(fast_ema, slope_lookback)
        slope_angle = math.degrees(math.atan(slope)) if slope != 0 else 0

        # EMA spread (distance between fast and slow)
        spread = current_fast - current_slow
        spread_pct = (spread / current_slow * 100) if current_slow > 0 else 0

        # Price position relative to EMAs
        above_fast = current_price > current_fast
        above_slow = current_price > current_slow

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        if crossover == "bullish":
            signal = "BULLISH"
            strength = 0.8
        elif crossover == "bearish":
            signal = "BEARISH"
            strength = 0.8
        elif current_fast > current_slow:
            # Uptrend
            signal = "BULLISH"
            # Strength based on spread and slope
            spread_strength = min(0.3, abs(spread_pct) / 3)
            slope_strength = min(0.3, abs(slope_angle) / 45)
            strength = 0.3 + spread_strength + slope_strength
            if above_fast and above_slow:
                strength = min(1.0, strength + 0.2)
        elif current_fast < current_slow:
            # Downtrend
            signal = "BEARISH"
            spread_strength = min(0.3, abs(spread_pct) / 3)
            slope_strength = min(0.3, abs(slope_angle) / 45)
            strength = 0.3 + spread_strength + slope_strength
            if not above_fast and not above_slow:
                strength = min(1.0, strength + 0.2)
        else:
            # EMAs converging
            strength = 0.1

        return IndicatorResult(
            name=self.name,
            value=round(current_fast, 6),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "fast_ema": round(current_fast, 6),
                "slow_ema": round(current_slow, 6),
                "spread": round(spread, 6),
                "spread_pct": round(spread_pct, 3),
                "slope_angle": round(slope_angle, 2),
                "crossover": crossover,
                "price_above_fast": above_fast,
                "price_above_slow": above_slow,
                "fast_period": fast_period,
                "slow_period": slow_period,
            }
        )

    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return []

        ema_values = []
        multiplier = 2 / (period + 1)

        # First EMA is SMA
        sma = sum(values[:period]) / period
        ema_values.append(sma)

        # Subsequent EMAs
        for i in range(period, len(values)):
            ema = (values[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)

        return ema_values

    def _calculate_slope(self, values: List[float], lookback: int) -> float:
        """
        Calculate normalized slope over lookback period.
        Returns slope as percentage change per period.
        """
        if len(values) < lookback + 1:
            return 0.0

        recent = values[-lookback:]
        first_val = recent[0]
        last_val = recent[-1]

        if first_val == 0:
            return 0.0

        # Percentage change per period
        total_change = (last_val - first_val) / first_val
        slope_per_period = total_change / lookback

        return slope_per_period


# Auto-register when module is imported
IndicatorRegistry.register(EMAIndicator)
