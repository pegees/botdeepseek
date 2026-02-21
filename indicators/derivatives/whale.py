"""Whale activity detection based on volume analysis."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class WhaleIndicator(IndicatorBase):
    """
    Whale activity detection indicator.

    Detects large trades and volume anomalies that may indicate
    smart money / institutional activity.

    Since we don't have direct trade data, we use candle volume
    analysis as a proxy:
    - Very high volume candles (> threshold * average)
    - Volume clusters (multiple high volume candles)
    - Direction of whale activity based on candle color

    Signals:
    - BULLISH: Large buying activity detected
    - BEARISH: Large selling activity detected
    - NEUTRAL: No significant whale activity
    """

    @property
    def name(self) -> str:
        return "whale"

    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback": 50,  # Candles to analyze
            "whale_threshold": 3.0,  # Volume must be 3x average to be "whale"
            "large_threshold": 2.0,  # Volume 2x average is "large"
            "cluster_window": 5,  # Candles to look for whale clusters
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Detect whale activity.

        Args:
            candles: List of OHLCV dicts
            params: Optional override for parameters

        Returns:
            IndicatorResult with whale analysis
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["lookback"]
        whale_threshold = p["whale_threshold"]
        large_threshold = p["large_threshold"]
        cluster_window = p["cluster_window"]

        if len(candles) < lookback:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        recent_candles = candles[-lookback:]
        volumes = [c["volume"] for c in recent_candles]
        avg_volume = sum(volumes) / len(volumes)

        if avg_volume == 0:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Zero average volume"}
            )

        # Analyze each candle for whale activity
        whale_candles = []
        large_candles = []

        for i, candle in enumerate(recent_candles):
            volume_ratio = candle["volume"] / avg_volume
            is_green = candle["close"] >= candle["open"]
            direction = "buy" if is_green else "sell"

            if volume_ratio >= whale_threshold:
                whale_candles.append({
                    "index": i,
                    "volume_ratio": volume_ratio,
                    "direction": direction,
                    "price": candle["close"],
                })
            elif volume_ratio >= large_threshold:
                large_candles.append({
                    "index": i,
                    "volume_ratio": volume_ratio,
                    "direction": direction,
                    "price": candle["close"],
                })

        # Check for recent whale activity (in last cluster_window candles)
        recent_whale_buys = 0
        recent_whale_sells = 0
        recent_large_buys = 0
        recent_large_sells = 0

        for wc in whale_candles:
            if wc["index"] >= len(recent_candles) - cluster_window:
                if wc["direction"] == "buy":
                    recent_whale_buys += 1
                else:
                    recent_whale_sells += 1

        for lc in large_candles:
            if lc["index"] >= len(recent_candles) - cluster_window:
                if lc["direction"] == "buy":
                    recent_large_buys += 1
                else:
                    recent_large_sells += 1

        # Calculate whale pressure
        total_whale_buys = sum(1 for w in whale_candles if w["direction"] == "buy")
        total_whale_sells = sum(1 for w in whale_candles if w["direction"] == "sell")

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        # Recent whale activity is most important
        if recent_whale_buys > 0 or recent_whale_sells > 0:
            if recent_whale_buys > recent_whale_sells:
                signal = "BULLISH"
                strength = 0.8 + min(0.2, recent_whale_buys * 0.1)
            elif recent_whale_sells > recent_whale_buys:
                signal = "BEARISH"
                strength = 0.8 + min(0.2, recent_whale_sells * 0.1)
            else:
                # Equal whale buys and sells - check large trades
                if recent_large_buys > recent_large_sells:
                    signal = "BULLISH"
                    strength = 0.5
                elif recent_large_sells > recent_large_buys:
                    signal = "BEARISH"
                    strength = 0.5
        elif recent_large_buys > 0 or recent_large_sells > 0:
            # No whale but large trades
            if recent_large_buys > recent_large_sells:
                signal = "BULLISH"
                strength = 0.4 + min(0.2, recent_large_buys * 0.1)
            elif recent_large_sells > recent_large_buys:
                signal = "BEARISH"
                strength = 0.4 + min(0.2, recent_large_sells * 0.1)
        else:
            # Check overall trend
            if total_whale_buys > total_whale_sells:
                signal = "BULLISH"
                strength = 0.3
            elif total_whale_sells > total_whale_buys:
                signal = "BEARISH"
                strength = 0.3

        return IndicatorResult(
            name=self.name,
            value=len(whale_candles),
            signal=signal,
            strength=round(min(1.0, strength), 2),
            metadata={
                "whale_candles": len(whale_candles),
                "large_candles": len(large_candles),
                "total_whale_buys": total_whale_buys,
                "total_whale_sells": total_whale_sells,
                "recent_whale_buys": recent_whale_buys,
                "recent_whale_sells": recent_whale_sells,
                "recent_large_buys": recent_large_buys,
                "recent_large_sells": recent_large_sells,
                "avg_volume": round(avg_volume, 2),
                "whale_threshold": whale_threshold,
            }
        )


# Auto-register when module is imported
IndicatorRegistry.register(WhaleIndicator)
