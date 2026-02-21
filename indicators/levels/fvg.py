"""Fair Value Gap (FVG) indicator."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


@dataclass
class FVG:
    """Represents a Fair Value Gap."""
    start_index: int
    top: float
    bottom: float
    type: str  # "bullish" or "bearish"
    filled: bool
    fill_percentage: float


class FVGIndicator(IndicatorBase):
    """
    Fair Value Gap (imbalance) detection.

    An FVG is a price imbalance where 3 consecutive candles leave a gap
    that wasn't traded through:
    - Bullish FVG: Gap up (candle 1 high < candle 3 low)
    - Bearish FVG: Gap down (candle 1 low > candle 3 high)

    Price often returns to fill these gaps.

    Signals:
    - BULLISH: Unfilled bullish FVG nearby (price may pull back to fill then continue up)
    - BEARISH: Unfilled bearish FVG nearby (price may pull back to fill then continue down)
    - NEUTRAL: No actionable FVG
    """

    @property
    def name(self) -> str:
        return "fvg"

    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback": 30,  # Candles to search for FVGs
            "min_gap_pct": 0.1,  # Minimum gap size as % of price
            "proximity_pct": 2.0,  # How close price must be to FVG for signal
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Detect and analyze Fair Value Gaps.

        Args:
            candles: List of OHLCV dicts
            params: Optional override for parameters

        Returns:
            IndicatorResult with FVG analysis
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["lookback"]
        min_gap_pct = p["min_gap_pct"]
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

        # Find all FVGs
        fvgs = []

        for i in range(len(recent_candles) - 2):
            candle1 = recent_candles[i]
            candle2 = recent_candles[i + 1]
            candle3 = recent_candles[i + 2]

            # Check for bullish FVG (gap up)
            if candle3["low"] > candle1["high"]:
                gap_size = candle3["low"] - candle1["high"]
                gap_pct = gap_size / candle2["close"] * 100

                if gap_pct >= min_gap_pct:
                    fvg = FVG(
                        start_index=i,
                        top=candle3["low"],
                        bottom=candle1["high"],
                        type="bullish",
                        filled=False,
                        fill_percentage=0.0
                    )
                    fvgs.append(fvg)

            # Check for bearish FVG (gap down)
            if candle3["high"] < candle1["low"]:
                gap_size = candle1["low"] - candle3["high"]
                gap_pct = gap_size / candle2["close"] * 100

                if gap_pct >= min_gap_pct:
                    fvg = FVG(
                        start_index=i,
                        top=candle1["low"],
                        bottom=candle3["high"],
                        type="bearish",
                        filled=False,
                        fill_percentage=0.0
                    )
                    fvgs.append(fvg)

        # Check fill status for each FVG
        for fvg in fvgs:
            for j in range(fvg.start_index + 3, len(recent_candles)):
                candle = recent_candles[j]

                if fvg.type == "bullish":
                    # Bullish FVG filled when price drops into gap
                    if candle["low"] <= fvg.top:
                        fill_depth = fvg.top - max(candle["low"], fvg.bottom)
                        gap_size = fvg.top - fvg.bottom
                        fvg.fill_percentage = min(1.0, fill_depth / gap_size) if gap_size > 0 else 0
                        if candle["low"] <= fvg.bottom:
                            fvg.filled = True
                            fvg.fill_percentage = 1.0
                            break

                else:  # bearish
                    # Bearish FVG filled when price rises into gap
                    if candle["high"] >= fvg.bottom:
                        fill_depth = min(candle["high"], fvg.top) - fvg.bottom
                        gap_size = fvg.top - fvg.bottom
                        fvg.fill_percentage = min(1.0, fill_depth / gap_size) if gap_size > 0 else 0
                        if candle["high"] >= fvg.top:
                            fvg.filled = True
                            fvg.fill_percentage = 1.0
                            break

        # Find unfilled FVGs near current price
        unfilled_bullish = []
        unfilled_bearish = []

        for fvg in fvgs:
            if fvg.filled:
                continue

            # Calculate distance to FVG
            if fvg.type == "bullish":
                # Bullish FVG is below current price, may get filled on pullback
                if current_price > fvg.top:
                    distance_pct = (current_price - fvg.top) / current_price * 100
                    if distance_pct <= proximity_pct:
                        unfilled_bullish.append((fvg, distance_pct))
            else:
                # Bearish FVG is above current price
                if current_price < fvg.bottom:
                    distance_pct = (fvg.bottom - current_price) / current_price * 100
                    if distance_pct <= proximity_pct:
                        unfilled_bearish.append((fvg, distance_pct))

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0
        nearest_fvg = None

        if unfilled_bullish and not unfilled_bearish:
            signal = "BULLISH"
            nearest = min(unfilled_bullish, key=lambda x: x[1])
            nearest_fvg = nearest[0]
            strength = 0.6 + (1 - nearest[1] / proximity_pct) * 0.3
        elif unfilled_bearish and not unfilled_bullish:
            signal = "BEARISH"
            nearest = min(unfilled_bearish, key=lambda x: x[1])
            nearest_fvg = nearest[0]
            strength = 0.6 + (1 - nearest[1] / proximity_pct) * 0.3
        elif unfilled_bullish and unfilled_bearish:
            # Both present - use nearer one
            bull_nearest = min(unfilled_bullish, key=lambda x: x[1])
            bear_nearest = min(unfilled_bearish, key=lambda x: x[1])

            if bull_nearest[1] < bear_nearest[1]:
                signal = "BULLISH"
                nearest_fvg = bull_nearest[0]
                strength = 0.5
            else:
                signal = "BEARISH"
                nearest_fvg = bear_nearest[0]
                strength = 0.5

        # Prepare FVG data for metadata
        fvg_data = []
        for fvg in fvgs[-5:]:  # Last 5 FVGs
            fvg_data.append({
                "type": fvg.type,
                "top": round(fvg.top, 6),
                "bottom": round(fvg.bottom, 6),
                "filled": fvg.filled,
                "fill_pct": round(fvg.fill_percentage * 100, 1)
            })

        return IndicatorResult(
            name=self.name,
            value=len([f for f in fvgs if not f.filled]),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "total_fvgs": len(fvgs),
                "unfilled_bullish": len(unfilled_bullish),
                "unfilled_bearish": len(unfilled_bearish),
                "nearest_fvg": {
                    "type": nearest_fvg.type,
                    "top": round(nearest_fvg.top, 6),
                    "bottom": round(nearest_fvg.bottom, 6),
                } if nearest_fvg else None,
                "recent_fvgs": fvg_data,
            }
        )


# Auto-register when module is imported
IndicatorRegistry.register(FVGIndicator)
