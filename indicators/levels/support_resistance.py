"""Support and Resistance level detection."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


@dataclass
class Level:
    """Represents a support or resistance level."""
    price: float
    type: str  # "support" or "resistance"
    touches: int
    strength: float


class SupportResistanceIndicator(IndicatorBase):
    """
    Support and Resistance level detection.

    Identifies key price levels based on:
    - Swing highs (resistance) and swing lows (support)
    - Number of touches (more touches = stronger level)
    - Proximity to current price

    Signals:
    - BULLISH: Price near strong support with bounce potential
    - BEARISH: Price near strong resistance with rejection potential
    - NEUTRAL: Price in no-man's land between levels
    """

    @property
    def name(self) -> str:
        return "support_resistance"

    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback": 50,  # Candles to analyze
            "tolerance_pct": 0.5,  # % tolerance for grouping levels
            "min_touches": 2,  # Minimum touches for valid level
            "proximity_pct": 1.0,  # % proximity to trigger signal
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Detect support and resistance levels.

        Args:
            candles: List of OHLCV dicts
            params: Optional override for parameters

        Returns:
            IndicatorResult with S/R analysis
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["lookback"]
        tolerance_pct = p["tolerance_pct"]
        min_touches = p["min_touches"]
        proximity_pct = p["proximity_pct"]

        if len(candles) < lookback:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        recent_candles = candles[-lookback:]
        current_price = recent_candles[-1]["close"]

        # Collect potential levels from swing points
        potential_levels = []

        # Find swing highs (resistance)
        for i in range(2, len(recent_candles) - 2):
            if self._is_swing_high(recent_candles, i):
                potential_levels.append({
                    "price": recent_candles[i]["high"],
                    "type": "resistance"
                })

        # Find swing lows (support)
        for i in range(2, len(recent_candles) - 2):
            if self._is_swing_low(recent_candles, i):
                potential_levels.append({
                    "price": recent_candles[i]["low"],
                    "type": "support"
                })

        if not potential_levels:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"levels": [], "error": "No levels found"}
            )

        # Cluster nearby levels
        clustered = self._cluster_levels(potential_levels, tolerance_pct)

        # Filter by minimum touches
        strong_levels = [l for l in clustered if l.touches >= min_touches]

        # Find nearest levels
        nearest_support = None
        nearest_resistance = None

        for level in strong_levels:
            if level.type == "support" and level.price < current_price:
                if nearest_support is None or level.price > nearest_support.price:
                    nearest_support = level
            elif level.type == "resistance" and level.price > current_price:
                if nearest_resistance is None or level.price < nearest_resistance.price:
                    nearest_resistance = level

        # Calculate distances
        support_distance_pct = None
        resistance_distance_pct = None

        if nearest_support:
            support_distance_pct = (current_price - nearest_support.price) / current_price * 100

        if nearest_resistance:
            resistance_distance_pct = (nearest_resistance.price - current_price) / current_price * 100

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        if support_distance_pct is not None and support_distance_pct <= proximity_pct:
            # Near support - potential bounce
            signal = "BULLISH"
            strength = min(1.0, nearest_support.strength * (1 - support_distance_pct / proximity_pct))
        elif resistance_distance_pct is not None and resistance_distance_pct <= proximity_pct:
            # Near resistance - potential rejection
            signal = "BEARISH"
            strength = min(1.0, nearest_resistance.strength * (1 - resistance_distance_pct / proximity_pct))

        # Prepare levels for metadata
        levels_data = [
            {
                "price": round(l.price, 6),
                "type": l.type,
                "touches": l.touches,
                "strength": round(l.strength, 2)
            }
            for l in strong_levels[:10]  # Top 10 levels
        ]

        return IndicatorResult(
            name=self.name,
            value=len(strong_levels),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "levels": levels_data,
                "nearest_support": round(nearest_support.price, 6) if nearest_support else None,
                "nearest_resistance": round(nearest_resistance.price, 6) if nearest_resistance else None,
                "support_distance_pct": round(support_distance_pct, 2) if support_distance_pct else None,
                "resistance_distance_pct": round(resistance_distance_pct, 2) if resistance_distance_pct else None,
            }
        )

    def _is_swing_high(self, candles: List[Dict], index: int) -> bool:
        """Check if candle at index is a swing high."""
        current = candles[index]["high"]
        return (
            candles[index - 2]["high"] < current and
            candles[index - 1]["high"] < current and
            candles[index + 1]["high"] < current and
            candles[index + 2]["high"] < current
        )

    def _is_swing_low(self, candles: List[Dict], index: int) -> bool:
        """Check if candle at index is a swing low."""
        current = candles[index]["low"]
        return (
            candles[index - 2]["low"] > current and
            candles[index - 1]["low"] > current and
            candles[index + 1]["low"] > current and
            candles[index + 2]["low"] > current
        )

    def _cluster_levels(self, levels: List[Dict], tolerance_pct: float) -> List[Level]:
        """Cluster nearby price levels together."""
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])

        clusters = []
        current_cluster = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            level = sorted_levels[i]
            prev_price = current_cluster[-1]["price"]

            # Check if within tolerance
            if abs(level["price"] - prev_price) / prev_price * 100 <= tolerance_pct:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                clusters.append(self._finalize_cluster(current_cluster))
                current_cluster = [level]

        # Finalize last cluster
        if current_cluster:
            clusters.append(self._finalize_cluster(current_cluster))

        return clusters

    def _finalize_cluster(self, cluster: List[Dict]) -> Level:
        """Convert cluster to a Level with average price and touch count."""
        avg_price = sum(l["price"] for l in cluster) / len(cluster)
        touches = len(cluster)

        # Determine type (majority wins)
        support_count = sum(1 for l in cluster if l["type"] == "support")
        resistance_count = len(cluster) - support_count

        level_type = "support" if support_count >= resistance_count else "resistance"

        # Strength based on touches
        strength = min(1.0, touches * 0.2)

        return Level(
            price=avg_price,
            type=level_type,
            touches=touches,
            strength=strength
        )


# Auto-register when module is imported
IndicatorRegistry.register(SupportResistanceIndicator)
