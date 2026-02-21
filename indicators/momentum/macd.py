"""MACD (Moving Average Convergence Divergence) indicator."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class MACDIndicator(IndicatorBase):
    """
    MACD indicator with histogram analysis and crossover detection.

    Signals:
    - BULLISH: MACD crosses above signal line, or histogram turning positive
    - BEARISH: MACD crosses below signal line, or histogram turning negative
    - NEUTRAL: No clear signal
    """

    @property
    def name(self) -> str:
        return "macd"

    def default_params(self) -> Dict[str, Any]:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Calculate MACD, signal line, and histogram.

        Args:
            candles: List of OHLCV dicts with 'close' key
            params: Optional override for fast_period, slow_period, signal_period

        Returns:
            IndicatorResult with MACD data and signal
        """
        p = {**self.default_params(), **(params or {})}
        fast_period = p["fast_period"]
        slow_period = p["slow_period"]
        signal_period = p["signal_period"]

        min_candles = slow_period + signal_period
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

        # MACD line = Fast EMA - Slow EMA
        # Align arrays (slow_ema is shorter)
        offset = len(fast_ema) - len(slow_ema)
        macd_line = [
            fast_ema[offset + i] - slow_ema[i]
            for i in range(len(slow_ema))
        ]

        if len(macd_line) < signal_period:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient MACD data"}
            )

        # Signal line = EMA of MACD line
        signal_line = self._calculate_ema(macd_line, signal_period)

        # Histogram = MACD - Signal
        hist_offset = len(macd_line) - len(signal_line)
        histogram = [
            macd_line[hist_offset + i] - signal_line[i]
            for i in range(len(signal_line))
        ]

        if len(histogram) < 2:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient histogram data"}
            )

        # Current values
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        prev_histogram = histogram[-2]

        # Detect crossovers
        crossover = None
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            macd_offset = len(macd_line) - len(signal_line)
            prev_macd = macd_line[-2 + macd_offset] if macd_offset == 0 else macd_line[-2]
            prev_signal = signal_line[-2]

            # Bullish crossover: MACD crosses above signal
            if prev_macd <= prev_signal and current_macd > current_signal:
                crossover = "bullish"
            # Bearish crossover: MACD crosses below signal
            elif prev_macd >= prev_signal and current_macd < current_signal:
                crossover = "bearish"

        # Detect histogram momentum shift
        histogram_shift = None
        if prev_histogram < 0 and current_histogram > 0:
            histogram_shift = "bullish"
        elif prev_histogram > 0 and current_histogram < 0:
            histogram_shift = "bearish"

        # Determine signal and strength
        signal = "NEUTRAL"
        strength = 0.0

        if crossover == "bullish" or histogram_shift == "bullish":
            signal = "BULLISH"
            # Strength based on histogram momentum
            if current_histogram > 0:
                strength = min(1.0, abs(current_histogram) / abs(current_macd) + 0.5 if current_macd != 0 else 0.7)
            else:
                strength = 0.6
        elif crossover == "bearish" or histogram_shift == "bearish":
            signal = "BEARISH"
            if current_histogram < 0:
                strength = min(1.0, abs(current_histogram) / abs(current_macd) + 0.5 if current_macd != 0 else 0.7)
            else:
                strength = 0.6
        else:
            # No crossover but check trend
            if current_histogram > 0 and current_histogram > prev_histogram:
                signal = "BULLISH"
                strength = 0.3
            elif current_histogram < 0 and current_histogram < prev_histogram:
                signal = "BEARISH"
                strength = 0.3
            else:
                strength = 0.1

        return IndicatorResult(
            name=self.name,
            value=round(current_macd, 6),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "macd": round(current_macd, 6),
                "signal_line": round(current_signal, 6),
                "histogram": round(current_histogram, 6),
                "prev_histogram": round(prev_histogram, 6),
                "crossover": crossover,
                "histogram_shift": histogram_shift,
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
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


# Auto-register when module is imported
IndicatorRegistry.register(MACDIndicator)
