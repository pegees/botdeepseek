"""CVD (Cumulative Volume Delta) indicator."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class CVDIndicator(IndicatorBase):
    """
    Cumulative Volume Delta indicator.

    Approximates buying/selling pressure by splitting candle volume
    based on close position within the candle range.

    CVD = cumulative sum of (buy volume - sell volume)

    Signals:
    - BULLISH: CVD rising while price stable/falling (accumulation) or CVD confirming uptrend
    - BEARISH: CVD falling while price stable/rising (distribution) or CVD confirming downtrend
    - NEUTRAL: CVD and price moving together normally
    """

    @property
    def name(self) -> str:
        return "cvd"

    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback": 20,  # Candles to calculate CVD over
            "divergence_threshold": 0.02,  # 2% price change threshold for divergence
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Calculate CVD and detect divergences.

        Args:
            candles: List of OHLCV dicts with 'open', 'high', 'low', 'close', 'volume' keys
            params: Optional override for lookback, divergence_threshold

        Returns:
            IndicatorResult with CVD data and signal
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["lookback"]
        div_threshold = p["divergence_threshold"]

        if len(candles) < lookback:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        # Calculate delta for each candle
        deltas = []
        for candle in candles[-lookback:]:
            delta = self._calculate_candle_delta(candle)
            deltas.append(delta)

        # Cumulative sum
        cvd_values = []
        cumulative = 0
        for delta in deltas:
            cumulative += delta
            cvd_values.append(cumulative)

        if len(cvd_values) < 5:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient CVD data"}
            )

        current_cvd = cvd_values[-1]
        prev_cvd = cvd_values[-5]  # Compare to 5 candles ago

        # CVD trend
        cvd_change = current_cvd - prev_cvd
        cvd_rising = cvd_change > 0

        # Price trend over same period
        recent_candles = candles[-lookback:]
        start_price = recent_candles[-5]["close"]
        end_price = recent_candles[-1]["close"]
        price_change_pct = (end_price - start_price) / start_price if start_price > 0 else 0
        price_rising = price_change_pct > 0

        # Detect divergence
        divergence = None
        if abs(price_change_pct) > div_threshold:
            # Bullish divergence: Price falling but CVD rising (accumulation)
            if not price_rising and cvd_rising:
                divergence = "bullish"
            # Bearish divergence: Price rising but CVD falling (distribution)
            elif price_rising and not cvd_rising:
                divergence = "bearish"

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        if divergence == "bullish":
            signal = "BULLISH"
            strength = 0.8
        elif divergence == "bearish":
            signal = "BEARISH"
            strength = 0.8
        elif cvd_rising and price_rising:
            # Confirmation of uptrend
            signal = "BULLISH"
            strength = 0.5
        elif not cvd_rising and not price_rising:
            # Confirmation of downtrend
            signal = "BEARISH"
            strength = 0.5
        else:
            # Mixed signals
            strength = 0.2
            if cvd_rising:
                signal = "BULLISH"
            else:
                signal = "BEARISH"

        # Calculate normalized CVD value (relative to total volume)
        total_volume = sum(c["volume"] for c in recent_candles)
        normalized_cvd = current_cvd / total_volume if total_volume > 0 else 0

        return IndicatorResult(
            name=self.name,
            value=round(normalized_cvd, 4),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "cvd": round(current_cvd, 2),
                "cvd_change": round(cvd_change, 2),
                "cvd_rising": cvd_rising,
                "price_change_pct": round(price_change_pct * 100, 2),
                "divergence": divergence,
                "normalized_cvd": round(normalized_cvd, 4),
                "lookback": lookback,
            }
        )

    def _calculate_candle_delta(self, candle: Dict) -> float:
        """
        Estimate buy/sell delta for a single candle.

        Uses the close position within the high-low range to split volume.
        If close is near high -> more buying, near low -> more selling.
        """
        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        volume = candle["volume"]

        # Avoid division by zero
        range_size = high - low
        if range_size == 0:
            # Doji candle - assume neutral
            return 0

        # Position of close within range (0 = at low, 1 = at high)
        position = (close - low) / range_size

        # Buy ratio (0.0 to 1.0)
        buy_ratio = position

        # Calculate delta
        buy_volume = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)
        delta = buy_volume - sell_volume

        return delta


# Auto-register when module is imported
IndicatorRegistry.register(CVDIndicator)
