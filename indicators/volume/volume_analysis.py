"""Volume analysis indicator with spike detection."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class VolumeIndicator(IndicatorBase):
    """
    Volume analysis indicator with spike detection and trend confirmation.

    Detects:
    - Volume spikes (current volume > threshold * MA)
    - Buying vs selling pressure based on candle direction
    - Volume trend (increasing/decreasing)

    Signals:
    - BULLISH: Volume spike on green candle (buying pressure)
    - BEARISH: Volume spike on red candle (selling pressure)
    - NEUTRAL: Normal volume or mixed signals
    """

    @property
    def name(self) -> str:
        return "volume"

    def default_params(self) -> Dict[str, Any]:
        return {
            "ma_period": 20,
            "spike_threshold": 2.0,  # 2x average = spike
            "high_volume_threshold": 1.5,  # 1.5x = high volume
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Calculate volume analysis.

        Args:
            candles: List of OHLCV dicts with 'open', 'close', 'volume' keys
            params: Optional override for ma_period, spike_threshold

        Returns:
            IndicatorResult with volume data and signal
        """
        p = {**self.default_params(), **(params or {})}
        ma_period = p["ma_period"]
        spike_threshold = p["spike_threshold"]
        high_volume_threshold = p["high_volume_threshold"]

        if len(candles) < ma_period:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        volumes = [c["volume"] for c in candles]
        current_volume = volumes[-1]

        # Calculate volume MA
        vol_ma = sum(volumes[-ma_period:]) / ma_period
        volume_ratio = current_volume / vol_ma if vol_ma > 0 else 1.0

        # Detect spike
        is_spike = volume_ratio >= spike_threshold
        is_high_volume = volume_ratio >= high_volume_threshold

        # Determine candle direction (buying vs selling)
        current_candle = candles[-1]
        is_green = current_candle["close"] >= current_candle["open"]

        # Volume trend (are volumes increasing over last 5 candles?)
        volume_trend = "neutral"
        if len(volumes) >= 5:
            recent_vols = volumes[-5:]
            if all(recent_vols[i] <= recent_vols[i + 1] for i in range(len(recent_vols) - 1)):
                volume_trend = "increasing"
            elif all(recent_vols[i] >= recent_vols[i + 1] for i in range(len(recent_vols) - 1)):
                volume_trend = "decreasing"

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        if is_spike:
            if is_green:
                signal = "BULLISH"
                strength = min(1.0, 0.5 + (volume_ratio - spike_threshold) * 0.2)
            else:
                signal = "BEARISH"
                strength = min(1.0, 0.5 + (volume_ratio - spike_threshold) * 0.2)
        elif is_high_volume:
            if is_green:
                signal = "BULLISH"
                strength = 0.4 + (volume_ratio - high_volume_threshold) * 0.2
            else:
                signal = "BEARISH"
                strength = 0.4 + (volume_ratio - high_volume_threshold) * 0.2
        else:
            # Normal volume - weak signal based on direction
            if is_green:
                signal = "BULLISH"
                strength = 0.2
            else:
                signal = "BEARISH"
                strength = 0.2

        # Adjust strength based on volume trend
        if volume_trend == "increasing":
            strength = min(1.0, strength + 0.1)

        return IndicatorResult(
            name=self.name,
            value=round(volume_ratio, 2),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "current_volume": current_volume,
                "volume_ma": round(vol_ma, 2),
                "volume_ratio": round(volume_ratio, 2),
                "is_spike": is_spike,
                "is_high_volume": is_high_volume,
                "candle_direction": "green" if is_green else "red",
                "volume_trend": volume_trend,
                "ma_period": ma_period,
                "spike_threshold": spike_threshold,
            }
        )


# Auto-register when module is imported
IndicatorRegistry.register(VolumeIndicator)
