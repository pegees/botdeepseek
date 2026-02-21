"""Market Structure indicator - HH/HL/LL/LH and Break of Structure detection."""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    index: int
    price: float
    type: str  # "high" or "low"


class MarketStructureIndicator(IndicatorBase):
    """
    Market Structure indicator for detecting:
    - Higher Highs (HH), Higher Lows (HL) = Uptrend
    - Lower Lows (LL), Lower Highs (LH) = Downtrend
    - Break of Structure (BOS) = Trend change signal

    Signals:
    - BULLISH: Uptrend structure (HH/HL) or bullish BOS
    - BEARISH: Downtrend structure (LL/LH) or bearish BOS
    - NEUTRAL: Ranging or unclear structure
    """

    @property
    def name(self) -> str:
        return "market_structure"

    def default_params(self) -> Dict[str, Any]:
        return {
            "swing_lookback": 3,  # Candles on each side to confirm swing point
            "min_swings": 4,  # Minimum swing points needed for analysis
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Analyze market structure.

        Args:
            candles: List of OHLCV dicts with 'high', 'low' keys
            params: Optional override for swing_lookback, min_swings

        Returns:
            IndicatorResult with structure analysis
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["swing_lookback"]
        min_swings = p["min_swings"]

        min_candles = lookback * 2 + min_swings
        if len(candles) < min_candles:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        # Find swing points
        swing_highs = self._find_swing_highs(candles, lookback)
        swing_lows = self._find_swing_lows(candles, lookback)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={
                    "error": "Insufficient swing points",
                    "swing_highs_count": len(swing_highs),
                    "swing_lows_count": len(swing_lows),
                }
            )

        # Analyze structure
        structure = self._analyze_structure(swing_highs, swing_lows)

        # Detect Break of Structure
        bos = self._detect_bos(candles, swing_highs, swing_lows)

        # Determine overall trend
        trend = structure["trend"]

        # Determine signal
        signal = "NEUTRAL"
        strength = 0.0

        if bos:
            # BOS takes priority
            if bos["type"] == "bullish":
                signal = "BULLISH"
                strength = 0.9
            else:
                signal = "BEARISH"
                strength = 0.9
        elif trend == "uptrend":
            signal = "BULLISH"
            strength = 0.6 + (structure["hh_count"] + structure["hl_count"]) * 0.05
        elif trend == "downtrend":
            signal = "BEARISH"
            strength = 0.6 + (structure["ll_count"] + structure["lh_count"]) * 0.05
        else:
            # Ranging
            strength = 0.2

        strength = min(1.0, strength)

        # Get recent swing points for metadata
        recent_swing_high = swing_highs[-1] if swing_highs else None
        recent_swing_low = swing_lows[-1] if swing_lows else None

        return IndicatorResult(
            name=self.name,
            value=1.0 if trend == "uptrend" else (-1.0 if trend == "downtrend" else 0.0),
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "trend": trend,
                "hh_count": structure["hh_count"],
                "hl_count": structure["hl_count"],
                "ll_count": structure["ll_count"],
                "lh_count": structure["lh_count"],
                "bos": bos,
                "recent_swing_high": recent_swing_high.price if recent_swing_high else None,
                "recent_swing_low": recent_swing_low.price if recent_swing_low else None,
                "swing_lookback": lookback,
            }
        )

    def _find_swing_highs(self, candles: List[Dict], lookback: int) -> List[SwingPoint]:
        """Find swing high points."""
        swing_highs = []

        for i in range(lookback, len(candles) - lookback):
            current_high = candles[i]["high"]
            is_swing_high = True

            # Check candles before
            for j in range(i - lookback, i):
                if candles[j]["high"] >= current_high:
                    is_swing_high = False
                    break

            # Check candles after
            if is_swing_high:
                for j in range(i + 1, i + lookback + 1):
                    if candles[j]["high"] >= current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append(SwingPoint(
                    index=i,
                    price=current_high,
                    type="high"
                ))

        return swing_highs

    def _find_swing_lows(self, candles: List[Dict], lookback: int) -> List[SwingPoint]:
        """Find swing low points."""
        swing_lows = []

        for i in range(lookback, len(candles) - lookback):
            current_low = candles[i]["low"]
            is_swing_low = True

            # Check candles before
            for j in range(i - lookback, i):
                if candles[j]["low"] <= current_low:
                    is_swing_low = False
                    break

            # Check candles after
            if is_swing_low:
                for j in range(i + 1, i + lookback + 1):
                    if candles[j]["low"] <= current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append(SwingPoint(
                    index=i,
                    price=current_low,
                    type="low"
                ))

        return swing_lows

    def _analyze_structure(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> Dict[str, Any]:
        """Analyze HH/HL/LL/LH pattern."""
        hh_count = 0  # Higher highs
        hl_count = 0  # Higher lows
        ll_count = 0  # Lower lows
        lh_count = 0  # Lower highs

        # Count higher highs and lower highs
        for i in range(1, len(swing_highs)):
            if swing_highs[i].price > swing_highs[i - 1].price:
                hh_count += 1
            else:
                lh_count += 1

        # Count higher lows and lower lows
        for i in range(1, len(swing_lows)):
            if swing_lows[i].price > swing_lows[i - 1].price:
                hl_count += 1
            else:
                ll_count += 1

        # Determine trend
        bullish_score = hh_count + hl_count
        bearish_score = ll_count + lh_count

        if bullish_score > bearish_score + 1:
            trend = "uptrend"
        elif bearish_score > bullish_score + 1:
            trend = "downtrend"
        else:
            trend = "ranging"

        return {
            "trend": trend,
            "hh_count": hh_count,
            "hl_count": hl_count,
            "ll_count": ll_count,
            "lh_count": lh_count,
        }

    def _detect_bos(
        self,
        candles: List[Dict],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect Break of Structure.

        Bullish BOS: Price breaks above previous swing high
        Bearish BOS: Price breaks below previous swing low
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        current_price = candles[-1]["close"]
        current_high = candles[-1]["high"]
        current_low = candles[-1]["low"]

        # Get the second-to-last swing points (the ones that could be broken)
        prev_swing_high = swing_highs[-2] if len(swing_highs) >= 2 else None
        prev_swing_low = swing_lows[-2] if len(swing_lows) >= 2 else None

        # Check for bullish BOS (break above previous swing high)
        if prev_swing_high and current_high > prev_swing_high.price:
            # Verify it's a clean break (close above, not just a wick)
            if current_price > prev_swing_high.price:
                return {
                    "type": "bullish",
                    "broken_level": prev_swing_high.price,
                    "current_price": current_price,
                }

        # Check for bearish BOS (break below previous swing low)
        if prev_swing_low and current_low < prev_swing_low.price:
            # Verify it's a clean break
            if current_price < prev_swing_low.price:
                return {
                    "type": "bearish",
                    "broken_level": prev_swing_low.price,
                    "current_price": current_price,
                }

        return None


# Auto-register when module is imported
IndicatorRegistry.register(MarketStructureIndicator)
