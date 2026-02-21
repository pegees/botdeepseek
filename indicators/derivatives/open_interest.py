"""Open Interest analysis indicator."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class OpenInterestIndicator(IndicatorBase):
    """
    Open Interest analysis indicator.

    Analyzes OI data when available to detect:
    - OI increasing with price = trend confirmation
    - OI increasing against price = potential reversal
    - OI decreasing = position closing, trend weakening

    Note: This indicator requires OI data to be provided in the candle metadata.
    If OI data is not available, it will return a neutral signal.

    Signals:
    - BULLISH: OI rising with price (longs accumulating) or OI falling with price (shorts covering)
    - BEARISH: OI rising against price (shorts accumulating) or OI falling against price (longs closing)
    - NEUTRAL: No clear signal or insufficient data
    """

    @property
    def name(self) -> str:
        return "open_interest"

    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback": 20,  # Candles to analyze
            "change_threshold_pct": 5.0,  # Minimum OI change % to be significant
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Analyze Open Interest data.

        Args:
            candles: List of OHLCV dicts, optionally with 'open_interest' key
            params: Optional override for parameters

        Returns:
            IndicatorResult with OI analysis
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["lookback"]
        change_threshold = p["change_threshold_pct"]

        if len(candles) < lookback:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        recent_candles = candles[-lookback:]

        # Check if OI data is available
        oi_values = []
        for c in recent_candles:
            oi = c.get("open_interest") or c.get("oi")
            if oi is not None:
                oi_values.append(float(oi))

        # If we don't have OI data, try to provide signal based on volume/price
        if len(oi_values) < 2:
            return self._fallback_analysis(recent_candles)

        # Calculate OI change
        start_oi = oi_values[0]
        end_oi = oi_values[-1]

        if start_oi == 0:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Zero starting OI"}
            )

        oi_change_pct = ((end_oi - start_oi) / start_oi) * 100

        # Calculate price change over same period
        start_price = recent_candles[0]["close"]
        end_price = recent_candles[-1]["close"]
        price_change_pct = ((end_price - start_price) / start_price) * 100

        # Determine signal based on OI/Price relationship
        signal = "NEUTRAL"
        strength = 0.0

        oi_increasing = oi_change_pct > change_threshold
        oi_decreasing = oi_change_pct < -change_threshold
        price_up = price_change_pct > 0
        price_down = price_change_pct < 0

        if oi_increasing:
            if price_up:
                # OI up + Price up = Longs accumulating, bullish
                signal = "BULLISH"
                strength = min(1.0, 0.5 + abs(oi_change_pct) / 20)
            else:
                # OI up + Price down = Shorts accumulating, bearish
                signal = "BEARISH"
                strength = min(1.0, 0.5 + abs(oi_change_pct) / 20)
        elif oi_decreasing:
            if price_up:
                # OI down + Price up = Short covering, bullish (but weakening)
                signal = "BULLISH"
                strength = min(0.6, 0.3 + abs(oi_change_pct) / 30)
            else:
                # OI down + Price down = Long liquidation, bearish (but weakening)
                signal = "BEARISH"
                strength = min(0.6, 0.3 + abs(oi_change_pct) / 30)
        else:
            # No significant OI change
            strength = 0.1

        return IndicatorResult(
            name=self.name,
            value=round(oi_change_pct, 2),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "oi_change_pct": round(oi_change_pct, 2),
                "price_change_pct": round(price_change_pct, 2),
                "start_oi": start_oi,
                "end_oi": end_oi,
                "oi_increasing": oi_increasing,
                "oi_decreasing": oi_decreasing,
                "data_points": len(oi_values),
            }
        )

    def _fallback_analysis(self, candles: List[Dict]) -> IndicatorResult:
        """
        Fallback analysis when OI data is not available.

        Uses volume and price action as a proxy for position changes.
        """
        # Calculate recent price trend
        if len(candles) < 5:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data", "fallback": True}
            )

        start_price = candles[0]["close"]
        end_price = candles[-1]["close"]
        price_change_pct = ((end_price - start_price) / start_price) * 100

        # Volume trend
        volumes = [c["volume"] for c in candles]
        vol_first_half = sum(volumes[:len(volumes)//2]) / (len(volumes)//2)
        vol_second_half = sum(volumes[len(volumes)//2:]) / (len(volumes) - len(volumes)//2)

        volume_increasing = vol_second_half > vol_first_half * 1.1

        # Estimate signal
        signal = "NEUTRAL"
        strength = 0.3

        if price_change_pct > 1 and volume_increasing:
            signal = "BULLISH"
            strength = 0.4
        elif price_change_pct < -1 and volume_increasing:
            signal = "BEARISH"
            strength = 0.4

        return IndicatorResult(
            name=self.name,
            value=0.0,
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "fallback": True,
                "oi_data_available": False,
                "price_change_pct": round(price_change_pct, 2),
                "volume_increasing": volume_increasing,
            }
        )


# Auto-register when module is imported
IndicatorRegistry.register(OpenInterestIndicator)
