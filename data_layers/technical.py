"""
Technical Analysis Data Layer
==============================
Full TA engine with 10+ indicators, each returning 0-100 scores.
Weight: 35% of confluence score.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from config import INDICATOR_PARAMS

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class IndicatorScore:
    """Score from a single indicator."""
    name: str
    score: float              # 0-100
    direction: SignalDirection
    value: float              # Raw indicator value
    description: str          # Human-readable explanation


@dataclass
class TechnicalScore:
    """Combined technical analysis score."""
    total_score: float        # 0-100 weighted average
    direction: SignalDirection
    indicator_scores: List[IndicatorScore]
    summary: str

    @property
    def is_bullish(self) -> bool:
        return self.direction == SignalDirection.LONG

    @property
    def is_bearish(self) -> bool:
        return self.direction == SignalDirection.SHORT


class TechnicalAnalyzer:
    """
    Technical analysis engine.
    Calculates 10+ indicators and combines into single 0-100 score.
    """

    # Indicator weights for combining (must sum to 1.0)
    INDICATOR_WEIGHTS = {
        "rsi": 0.12,
        "macd": 0.12,
        "ema_alignment": 0.10,
        "bollinger": 0.10,
        "stoch_rsi": 0.10,
        "volume": 0.10,
        "obv": 0.08,
        "vwap": 0.08,
        "support_resistance": 0.10,
        "pattern": 0.10,
    }

    def __init__(self):
        # Validate weights sum to 1.0
        total = sum(self.INDICATOR_WEIGHTS.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Indicator weights must sum to 1.0, got {total}")

    def calculate_rsi(self, closes: List[float]) -> IndicatorScore:
        """
        RSI (Relative Strength Index).
        Score: 100 at RSI 20 (oversold), 0 at RSI 80 (overbought).
        """
        period = INDICATOR_PARAMS.rsi_period

        if len(closes) < period + 1:
            return IndicatorScore("RSI", 50, SignalDirection.NEUTRAL, 50, "Insufficient data")

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Convert RSI to score
        # RSI < 30 = bullish (high score), RSI > 70 = bearish (low score)
        if rsi <= INDICATOR_PARAMS.rsi_oversold:
            score = 80 + (INDICATOR_PARAMS.rsi_oversold - rsi) * 2  # 80-100
            direction = SignalDirection.LONG
            desc = f"Oversold ({rsi:.0f})"
        elif rsi >= INDICATOR_PARAMS.rsi_overbought:
            score = 20 - (rsi - INDICATOR_PARAMS.rsi_overbought) * 2  # 0-20
            direction = SignalDirection.SHORT
            desc = f"Overbought ({rsi:.0f})"
        else:
            # Neutral zone - score based on distance from 50
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = f"Neutral ({rsi:.0f})"

        return IndicatorScore("RSI", max(0, min(100, score)), direction, rsi, desc)

    def calculate_macd(self, closes: List[float]) -> IndicatorScore:
        """
        MACD (Moving Average Convergence Divergence).
        Score based on MACD line vs signal line and histogram direction.
        """
        fast = INDICATOR_PARAMS.macd_fast
        slow = INDICATOR_PARAMS.macd_slow
        signal_period = INDICATOR_PARAMS.macd_signal

        if len(closes) < slow + signal_period:
            return IndicatorScore("MACD", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        closes_arr = np.array(closes)

        # Calculate EMAs
        ema_fast = self._ema(closes_arr, fast)
        ema_slow = self._ema(closes_arr, slow)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = self._ema(macd_line, signal_period)

        # Histogram
        histogram = macd_line - signal_line

        # Current values
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0

        # Score calculation
        # Bullish: MACD > signal, histogram positive and increasing
        # Bearish: MACD < signal, histogram negative and decreasing

        score = 50
        if current_macd > current_signal:
            score += 25
            if current_hist > prev_hist:
                score += 25  # Momentum increasing
            direction = SignalDirection.LONG
            desc = "Bullish crossover" if current_hist > 0 else "Turning bullish"
        elif current_macd < current_signal:
            score -= 25
            if current_hist < prev_hist:
                score -= 25  # Momentum decreasing
            direction = SignalDirection.SHORT
            desc = "Bearish crossover" if current_hist < 0 else "Turning bearish"
        else:
            direction = SignalDirection.NEUTRAL
            desc = "Neutral"

        return IndicatorScore("MACD", max(0, min(100, score)), direction, current_macd, desc)

    def calculate_ema_alignment(self, closes: List[float]) -> IndicatorScore:
        """
        EMA Alignment (9, 21, 50, 200).
        Perfect bullish: 9 > 21 > 50 > 200.
        """
        if len(closes) < INDICATOR_PARAMS.ema_trend:
            return IndicatorScore("EMA", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        closes_arr = np.array(closes)

        ema_9 = self._ema(closes_arr, INDICATOR_PARAMS.ema_fast)[-1]
        ema_21 = self._ema(closes_arr, INDICATOR_PARAMS.ema_medium)[-1]
        ema_50 = self._ema(closes_arr, INDICATOR_PARAMS.ema_slow)[-1]
        ema_200 = self._ema(closes_arr, INDICATOR_PARAMS.ema_trend)[-1]

        current_price = closes[-1]

        # Count bullish alignments
        bullish_count = 0
        if ema_9 > ema_21:
            bullish_count += 1
        if ema_21 > ema_50:
            bullish_count += 1
        if ema_50 > ema_200:
            bullish_count += 1
        if current_price > ema_9:
            bullish_count += 1

        # Score: 4 bullish = 100, 0 bullish = 0
        score = bullish_count * 25

        if bullish_count >= 3:
            direction = SignalDirection.LONG
            desc = f"Bullish alignment ({bullish_count}/4)"
        elif bullish_count <= 1:
            direction = SignalDirection.SHORT
            desc = f"Bearish alignment ({bullish_count}/4)"
        else:
            direction = SignalDirection.NEUTRAL
            desc = f"Mixed alignment ({bullish_count}/4)"

        return IndicatorScore("EMA", score, direction, ema_21, desc)

    def calculate_bollinger(self, closes: List[float]) -> IndicatorScore:
        """
        Bollinger Bands.
        Score based on price position relative to bands.
        """
        period = INDICATOR_PARAMS.bb_period
        std_mult = INDICATOR_PARAMS.bb_std

        if len(closes) < period:
            return IndicatorScore("BB", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        closes_arr = np.array(closes[-period:])
        sma = np.mean(closes_arr)
        std = np.std(closes_arr)

        upper = sma + (std_mult * std)
        lower = sma - (std_mult * std)

        current = closes[-1]

        # Calculate %B (position within bands)
        band_width = upper - lower
        if band_width == 0:
            pct_b = 0.5
        else:
            pct_b = (current - lower) / band_width

        # Score: Near lower band (0) = bullish, near upper (1) = bearish
        if pct_b <= 0.2:
            score = 80 + (0.2 - pct_b) * 100
            direction = SignalDirection.LONG
            desc = f"Near lower band ({pct_b:.2f})"
        elif pct_b >= 0.8:
            score = 20 - (pct_b - 0.8) * 100
            direction = SignalDirection.SHORT
            desc = f"Near upper band ({pct_b:.2f})"
        else:
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = f"Mid-band ({pct_b:.2f})"

        return IndicatorScore("BB", max(0, min(100, score)), direction, pct_b, desc)

    def calculate_stoch_rsi(self, closes: List[float]) -> IndicatorScore:
        """
        Stochastic RSI.
        Combines RSI with Stochastic oscillator.
        """
        period = INDICATOR_PARAMS.stoch_rsi_period
        k_period = INDICATOR_PARAMS.stoch_rsi_k

        if len(closes) < period * 2:
            return IndicatorScore("StochRSI", 50, SignalDirection.NEUTRAL, 50, "Insufficient data")

        # Calculate RSI first
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate RSI values over window
        rsi_values = []
        for i in range(period, len(closes)):
            avg_gain = np.mean(gains[i-period:i])
            avg_loss = np.mean(losses[i-period:i])
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        if len(rsi_values) < period:
            return IndicatorScore("StochRSI", 50, SignalDirection.NEUTRAL, 50, "Insufficient data")

        # Stochastic of RSI
        rsi_arr = np.array(rsi_values[-period:])
        rsi_min = np.min(rsi_arr)
        rsi_max = np.max(rsi_arr)

        if rsi_max == rsi_min:
            stoch_rsi = 50
        else:
            stoch_rsi = (rsi_values[-1] - rsi_min) / (rsi_max - rsi_min) * 100

        # Score: <20 = oversold (bullish), >80 = overbought (bearish)
        if stoch_rsi <= 20:
            score = 80 + (20 - stoch_rsi)
            direction = SignalDirection.LONG
            desc = f"Oversold ({stoch_rsi:.0f})"
        elif stoch_rsi >= 80:
            score = 20 - (stoch_rsi - 80)
            direction = SignalDirection.SHORT
            desc = f"Overbought ({stoch_rsi:.0f})"
        else:
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = f"Neutral ({stoch_rsi:.0f})"

        return IndicatorScore("StochRSI", max(0, min(100, score)), direction, stoch_rsi, desc)

    def calculate_volume(self, volumes: List[float], closes: List[float]) -> IndicatorScore:
        """
        Volume analysis.
        Score based on volume spike with price direction.
        """
        period = INDICATOR_PARAMS.volume_ma_period

        if len(volumes) < period:
            return IndicatorScore("Volume", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        avg_volume = np.mean(volumes[-period:-1])
        current_volume = volumes[-1]

        if avg_volume == 0:
            volume_ratio = 1
        else:
            volume_ratio = current_volume / avg_volume

        # Price direction
        price_change = closes[-1] - closes[-2] if len(closes) > 1 else 0

        # High volume + price up = bullish, high volume + price down = bearish
        spike = volume_ratio >= INDICATOR_PARAMS.volume_spike_multiplier

        if spike and price_change > 0:
            score = 70 + min(30, (volume_ratio - 2) * 10)
            direction = SignalDirection.LONG
            desc = f"Volume spike ({volume_ratio:.1f}x) + up"
        elif spike and price_change < 0:
            score = 30 - min(30, (volume_ratio - 2) * 10)
            direction = SignalDirection.SHORT
            desc = f"Volume spike ({volume_ratio:.1f}x) + down"
        else:
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = f"Normal volume ({volume_ratio:.1f}x)"

        return IndicatorScore("Volume", max(0, min(100, score)), direction, volume_ratio, desc)

    def calculate_obv(self, closes: List[float], volumes: List[float]) -> IndicatorScore:
        """
        On-Balance Volume.
        Trend confirmation through volume flow.
        """
        if len(closes) < INDICATOR_PARAMS.obv_ma_period:
            return IndicatorScore("OBV", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        # Calculate OBV
        obv = [0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        obv_arr = np.array(obv)
        obv_ma = self._sma(obv_arr, INDICATOR_PARAMS.obv_ma_period)

        current_obv = obv_arr[-1]
        current_ma = obv_ma[-1]

        # OBV above MA = bullish, below = bearish
        if current_obv > current_ma * 1.05:
            score = 75
            direction = SignalDirection.LONG
            desc = "OBV above MA (accumulation)"
        elif current_obv < current_ma * 0.95:
            score = 25
            direction = SignalDirection.SHORT
            desc = "OBV below MA (distribution)"
        else:
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = "OBV neutral"

        return IndicatorScore("OBV", score, direction, current_obv, desc)

    def calculate_vwap(
        self, highs: List[float], lows: List[float],
        closes: List[float], volumes: List[float]
    ) -> IndicatorScore:
        """
        VWAP (Volume Weighted Average Price).
        Price vs VWAP for intraday bias.
        """
        if len(closes) < 10:
            return IndicatorScore("VWAP", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        # Typical price
        typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]

        # VWAP
        cum_tp_vol = np.cumsum([tp * v for tp, v in zip(typical, volumes)])
        cum_vol = np.cumsum(volumes)

        vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else closes[-1]
        current_price = closes[-1]

        # Distance from VWAP
        distance_pct = (current_price - vwap) / vwap * 100

        if distance_pct < -1:
            score = 70 + min(30, abs(distance_pct) * 5)
            direction = SignalDirection.LONG
            desc = f"Below VWAP ({distance_pct:.1f}%)"
        elif distance_pct > 1:
            score = 30 - min(30, distance_pct * 5)
            direction = SignalDirection.SHORT
            desc = f"Above VWAP ({distance_pct:.1f}%)"
        else:
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = f"At VWAP ({distance_pct:.1f}%)"

        return IndicatorScore("VWAP", max(0, min(100, score)), direction, vwap, desc)

    def calculate_support_resistance(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> IndicatorScore:
        """
        Support/Resistance levels.
        Score based on proximity to key levels.
        """
        lookback = INDICATOR_PARAMS.sr_lookback

        if len(closes) < lookback:
            return IndicatorScore("S/R", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        highs_arr = np.array(highs[-lookback:])
        lows_arr = np.array(lows[-lookback:])
        current = closes[-1]

        # Find significant levels
        resistance = np.max(highs_arr)
        support = np.min(lows_arr)

        price_range = resistance - support
        if price_range == 0:
            return IndicatorScore("S/R", 50, SignalDirection.NEUTRAL, 0, "No range")

        # Position within range
        position = (current - support) / price_range

        # Near support = bullish, near resistance = bearish
        if position <= 0.2:
            score = 80
            direction = SignalDirection.LONG
            desc = f"Near support (${support:.2f})"
        elif position >= 0.8:
            score = 20
            direction = SignalDirection.SHORT
            desc = f"Near resistance (${resistance:.2f})"
        else:
            score = 50
            direction = SignalDirection.NEUTRAL
            desc = f"Mid-range ({position:.0%})"

        return IndicatorScore("S/R", score, direction, current, desc)

    def calculate_pattern(self, opens: List[float], highs: List[float],
                         lows: List[float], closes: List[float]) -> IndicatorScore:
        """
        Candlestick pattern recognition.
        Detects bullish/bearish patterns.
        """
        if len(closes) < 5:
            return IndicatorScore("Pattern", 50, SignalDirection.NEUTRAL, 0, "Insufficient data")

        # Recent candles
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        prev_c = closes[-2]

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            return IndicatorScore("Pattern", 50, SignalDirection.NEUTRAL, 0, "Doji")

        body_pct = body / total_range

        patterns_found = []
        score = 50
        direction = SignalDirection.NEUTRAL

        # Hammer (bullish)
        if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
            patterns_found.append("Hammer")
            score = 75
            direction = SignalDirection.LONG

        # Inverted Hammer (bullish)
        if upper_wick > body * 2 and lower_wick < body * 0.5 and c > prev_c:
            patterns_found.append("Inv Hammer")
            score = 70
            direction = SignalDirection.LONG

        # Shooting Star (bearish)
        if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
            patterns_found.append("Shooting Star")
            score = 25
            direction = SignalDirection.SHORT

        # Engulfing
        prev_o, prev_h, prev_l, prev_c2 = opens[-2], highs[-2], lows[-2], closes[-2]
        prev_body = abs(prev_c2 - prev_o)

        if c > o and prev_c2 < prev_o:  # Bullish engulfing
            if body > prev_body and o < prev_c2 and c > prev_o:
                patterns_found.append("Bull Engulf")
                score = 80
                direction = SignalDirection.LONG

        if c < o and prev_c2 > prev_o:  # Bearish engulfing
            if body > prev_body and o > prev_c2 and c < prev_o:
                patterns_found.append("Bear Engulf")
                score = 20
                direction = SignalDirection.SHORT

        # Doji
        if body_pct < 0.1:
            patterns_found.append("Doji")
            # Doji is neutral, but context matters
            score = 50

        desc = ", ".join(patterns_found) if patterns_found else "No pattern"

        return IndicatorScore("Pattern", score, direction, 0, desc)

    def analyze(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> TechnicalScore:
        """
        Run all indicators and return combined score.
        """
        scores: List[IndicatorScore] = []

        # Calculate all indicators
        scores.append(self.calculate_rsi(closes))
        scores.append(self.calculate_macd(closes))
        scores.append(self.calculate_ema_alignment(closes))
        scores.append(self.calculate_bollinger(closes))
        scores.append(self.calculate_stoch_rsi(closes))
        scores.append(self.calculate_volume(volumes, closes))
        scores.append(self.calculate_obv(closes, volumes))
        scores.append(self.calculate_vwap(highs, lows, closes, volumes))
        scores.append(self.calculate_support_resistance(highs, lows, closes))
        scores.append(self.calculate_pattern(opens, highs, lows, closes))

        # Calculate weighted average
        total_score = 0
        for score in scores:
            weight = self.INDICATOR_WEIGHTS.get(score.name.lower(), 0.1)
            total_score += score.score * weight

        # Determine overall direction
        long_count = sum(1 for s in scores if s.direction == SignalDirection.LONG)
        short_count = sum(1 for s in scores if s.direction == SignalDirection.SHORT)

        if long_count >= 6:
            direction = SignalDirection.LONG
        elif short_count >= 6:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Create summary
        strong_signals = [s for s in scores if s.score >= 70 or s.score <= 30]
        summary_parts = [f"{s.name}: {s.description}" for s in strong_signals[:3]]
        summary = " | ".join(summary_parts) if summary_parts else "Mixed signals"

        return TechnicalScore(
            total_score=round(total_score, 1),
            direction=direction,
            indicator_scores=scores,
            summary=summary
        )

    # Helper functions
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA."""
        return np.convolve(data, np.ones(period)/period, mode='valid')


# Singleton
_analyzer: Optional[TechnicalAnalyzer] = None


def get_technical_analyzer() -> TechnicalAnalyzer:
    """Get the analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TechnicalAnalyzer()
    return _analyzer
