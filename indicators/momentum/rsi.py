"""RSI (Relative Strength Index) indicator with divergence detection."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class RSIIndicator(IndicatorBase):
    """
    RSI indicator with overbought/oversold detection and divergence analysis.

    Signals:
    - BULLISH: RSI < oversold (default 30) or bullish divergence
    - BEARISH: RSI > overbought (default 70) or bearish divergence
    - NEUTRAL: RSI between thresholds
    """

    @property
    def name(self) -> str:
        return "rsi"

    def default_params(self) -> Dict[str, Any]:
        return {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Calculate RSI and detect signals.

        Args:
            candles: List of OHLCV dicts with 'close' key
            params: Optional override for period, overbought, oversold

        Returns:
            IndicatorResult with RSI value and signal
        """
        p = {**self.default_params(), **(params or {})}
        period = p["period"]
        overbought = p["overbought"]
        oversold = p["oversold"]

        if len(candles) < period + 1:
            return IndicatorResult(
                name=self.name,
                value=50.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        # Extract closes
        closes = [c["close"] for c in candles]

        # Calculate price changes
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # Calculate RSI using Wilder's smoothing
        rsi_values = self._calculate_rsi_series(changes, period)

        if not rsi_values:
            return IndicatorResult(
                name=self.name,
                value=50.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "RSI calculation failed"}
            )

        current_rsi = rsi_values[-1]

        # Detect divergence if we have enough RSI values
        divergence = None
        if len(rsi_values) >= 5:
            divergence = self._detect_divergence(closes[-(len(rsi_values)):], rsi_values)

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        if divergence == "bullish":
            signal = "BULLISH"
            strength = 0.8
        elif divergence == "bearish":
            signal = "BEARISH"
            strength = 0.8
        elif current_rsi <= oversold:
            signal = "BULLISH"
            # Strength increases as RSI gets more oversold
            strength = min(1.0, (oversold - current_rsi) / oversold + 0.5)
        elif current_rsi >= overbought:
            signal = "BEARISH"
            # Strength increases as RSI gets more overbought
            strength = min(1.0, (current_rsi - overbought) / (100 - overbought) + 0.5)
        else:
            # Neutral but calculate relative strength
            if current_rsi > 50:
                strength = (current_rsi - 50) / 50 * 0.3
            else:
                strength = (50 - current_rsi) / 50 * 0.3

        return IndicatorResult(
            name=self.name,
            value=round(current_rsi, 2),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "period": period,
                "overbought": overbought,
                "oversold": oversold,
                "divergence": divergence,
                "previous_rsi": round(rsi_values[-2], 2) if len(rsi_values) >= 2 else None,
            }
        )

    def _calculate_rsi_series(self, changes: List[float], period: int) -> List[float]:
        """Calculate RSI series using Wilder's smoothing method."""
        if len(changes) < period:
            return []

        rsi_values = []

        # Initial averages (simple average for first period)
        gains = [max(0, c) for c in changes[:period]]
        losses = [abs(min(0, c)) for c in changes[:period]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # First RSI value
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

        # Subsequent values using Wilder's smoothing
        for i in range(period, len(changes)):
            change = changes[i]
            gain = max(0, change)
            loss = abs(min(0, change))

            # Wilder's smoothing
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    def _detect_divergence(self, prices: List[float], rsi_values: List[float]) -> Optional[str]:
        """
        Detect bullish or bearish divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        if len(prices) < 5 or len(rsi_values) < 5:
            return None

        # Look at recent swing points (simplified)
        # Compare last 5 values for divergence pattern
        recent_prices = prices[-5:]
        recent_rsi = rsi_values[-5:]

        # Find local extremes
        price_lows = []
        price_highs = []
        rsi_lows = []
        rsi_highs = []

        for i in range(1, len(recent_prices) - 1):
            # Local low
            if recent_prices[i] < recent_prices[i - 1] and recent_prices[i] < recent_prices[i + 1]:
                price_lows.append((i, recent_prices[i]))
                rsi_lows.append((i, recent_rsi[i]))
            # Local high
            if recent_prices[i] > recent_prices[i - 1] and recent_prices[i] > recent_prices[i + 1]:
                price_highs.append((i, recent_prices[i]))
                rsi_highs.append((i, recent_rsi[i]))

        # Check for bullish divergence (price lower low, RSI higher low)
        if len(price_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                return "bullish"

        # Check for bearish divergence (price higher high, RSI lower high)
        if len(price_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                return "bearish"

        return None


# Auto-register when module is imported
IndicatorRegistry.register(RSIIndicator)
