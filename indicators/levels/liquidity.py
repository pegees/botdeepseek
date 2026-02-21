"""Liquidity Sweep detection indicator."""
from typing import Dict, List, Any, Optional
from ..base import IndicatorBase, IndicatorResult
from ..registry import IndicatorRegistry


class LiquiditySweepIndicator(IndicatorBase):
    """
    Liquidity Sweep (stop hunt) detection.

    A liquidity sweep occurs when price briefly breaks beyond a key level
    (taking out stop losses) then reverses back - often indicating
    smart money manipulation before a move in the opposite direction.

    Detects:
    - Wicks beyond recent swing highs/lows that close back inside
    - Higher timeframe level sweeps
    - Equal highs/lows being swept

    Signals:
    - BULLISH: Sweep of lows followed by reversal (potential long)
    - BEARISH: Sweep of highs followed by reversal (potential short)
    - NEUTRAL: No recent sweep detected
    """

    @property
    def name(self) -> str:
        return "liquidity_sweep"

    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback": 20,  # Candles to find levels to sweep
            "sweep_candles": 3,  # Recent candles to check for sweeps
            "min_wick_ratio": 0.5,  # Min wick size relative to body for valid sweep
        }

    def calculate(self, candles: List[Dict], params: Optional[Dict] = None) -> IndicatorResult:
        """
        Detect liquidity sweeps.

        Args:
            candles: List of OHLCV dicts
            params: Optional override for parameters

        Returns:
            IndicatorResult with sweep analysis
        """
        p = {**self.default_params(), **(params or {})}
        lookback = p["lookback"]
        sweep_candles = p["sweep_candles"]
        min_wick_ratio = p["min_wick_ratio"]

        if len(candles) < lookback + sweep_candles:
            return IndicatorResult(
                name=self.name,
                value=0.0,
                signal="NEUTRAL",
                strength=0.0,
                metadata={"error": "Insufficient data"}
            )

        # Find key levels from earlier candles (potential liquidity)
        history = candles[-(lookback + sweep_candles):-sweep_candles]
        recent = candles[-sweep_candles:]

        # Find swing lows and highs in history
        swing_lows = []
        swing_highs = []

        for i in range(1, len(history) - 1):
            if history[i]["low"] < history[i-1]["low"] and history[i]["low"] < history[i+1]["low"]:
                swing_lows.append(history[i]["low"])
            if history[i]["high"] > history[i-1]["high"] and history[i]["high"] > history[i+1]["high"]:
                swing_highs.append(history[i]["high"])

        # Also check for equal highs/lows (obvious liquidity pools)
        recent_lows = [c["low"] for c in history[-5:]]
        recent_highs = [c["high"] for c in history[-5:]]

        equal_lows = self._find_equal_levels(recent_lows)
        equal_highs = self._find_equal_levels(recent_highs)

        if equal_lows:
            swing_lows.extend(equal_lows)
        if equal_highs:
            swing_highs.extend(equal_highs)

        # Check recent candles for sweeps
        bullish_sweep = None
        bearish_sweep = None

        for candle in recent:
            # Check for low sweep (bullish setup)
            for level in swing_lows:
                if self._is_low_sweep(candle, level, min_wick_ratio):
                    bullish_sweep = {
                        "type": "bullish",
                        "swept_level": level,
                        "candle_low": candle["low"],
                        "candle_close": candle["close"],
                    }

            # Check for high sweep (bearish setup)
            for level in swing_highs:
                if self._is_high_sweep(candle, level, min_wick_ratio):
                    bearish_sweep = {
                        "type": "bearish",
                        "swept_level": level,
                        "candle_high": candle["high"],
                        "candle_close": candle["close"],
                    }

        # Determine signal (most recent sweep wins)
        signal = "NEUTRAL"
        strength = 0.0
        sweep = None

        if bullish_sweep and not bearish_sweep:
            signal = "BULLISH"
            sweep = bullish_sweep
            strength = 0.8
        elif bearish_sweep and not bullish_sweep:
            signal = "BEARISH"
            sweep = bearish_sweep
            strength = 0.8
        elif bullish_sweep and bearish_sweep:
            # Both detected - use the stronger one based on wick size
            bullish_wick = abs(bullish_sweep["swept_level"] - bullish_sweep["candle_close"])
            bearish_wick = abs(bearish_sweep["swept_level"] - bearish_sweep["candle_close"])

            if bullish_wick > bearish_wick:
                signal = "BULLISH"
                sweep = bullish_sweep
                strength = 0.7
            else:
                signal = "BEARISH"
                sweep = bearish_sweep
                strength = 0.7

        return IndicatorResult(
            name=self.name,
            value=1.0 if sweep else 0.0,
            signal=signal,
            strength=round(strength, 2),
            metadata={
                "sweep_detected": sweep is not None,
                "sweep": sweep,
                "swing_lows_found": len(swing_lows),
                "swing_highs_found": len(swing_highs),
                "equal_lows": equal_lows,
                "equal_highs": equal_highs,
            }
        )

    def _is_low_sweep(self, candle: Dict, level: float, min_wick_ratio: float) -> bool:
        """
        Check if candle swept a low level then reversed.

        Conditions:
        - Candle low went below the level
        - Candle closed above the level
        - Lower wick is significant
        """
        if candle["low"] >= level:
            return False

        if candle["close"] <= level:
            return False

        # Check wick ratio
        body = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]

        if body == 0:
            return lower_wick > 0

        return lower_wick / body >= min_wick_ratio

    def _is_high_sweep(self, candle: Dict, level: float, min_wick_ratio: float) -> bool:
        """
        Check if candle swept a high level then reversed.

        Conditions:
        - Candle high went above the level
        - Candle closed below the level
        - Upper wick is significant
        """
        if candle["high"] <= level:
            return False

        if candle["close"] >= level:
            return False

        # Check wick ratio
        body = abs(candle["close"] - candle["open"])
        upper_wick = candle["high"] - max(candle["open"], candle["close"])

        if body == 0:
            return upper_wick > 0

        return upper_wick / body >= min_wick_ratio

    def _find_equal_levels(self, prices: List[float], tolerance_pct: float = 0.1) -> List[float]:
        """Find equal highs or lows (within tolerance)."""
        if len(prices) < 2:
            return []

        equal_levels = []
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                avg = (prices[i] + prices[j]) / 2
                diff_pct = abs(prices[i] - prices[j]) / avg * 100

                if diff_pct <= tolerance_pct:
                    equal_levels.append(avg)

        return list(set(equal_levels))


# Auto-register when module is imported
IndicatorRegistry.register(LiquiditySweepIndicator)
