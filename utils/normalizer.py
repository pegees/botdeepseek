"""
Score Normalization Functions
==============================
Utility functions for normalizing indicator values to 0-100 scores.
"""
from typing import Tuple


def normalize_to_score(
    value: float,
    min_val: float,
    max_val: float,
    invert: bool = False
) -> float:
    """
    Normalize a value to 0-100 score.

    Args:
        value: The raw value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        invert: If True, lower values get higher scores

    Returns:
        Score between 0 and 100
    """
    if max_val == min_val:
        return 50.0

    # Clamp to range
    clamped = max(min_val, min(max_val, value))

    # Normalize to 0-1
    normalized = (clamped - min_val) / (max_val - min_val)

    # Convert to 0-100
    score = normalized * 100

    if invert:
        score = 100 - score

    return round(score, 1)


def rsi_to_score(rsi: float, direction: str = "long") -> Tuple[float, str]:
    """
    Convert RSI value to score based on direction.

    For longs:
        RSI < 30 = oversold = high score (bullish)
        RSI > 70 = overbought = low score

    For shorts:
        RSI > 70 = overbought = high score (bearish)
        RSI < 30 = oversold = low score
    """
    if direction == "long":
        if rsi <= 20:
            return 95, "Extreme oversold"
        elif rsi <= 30:
            return 85, "Oversold"
        elif rsi <= 40:
            return 70, "Approaching oversold"
        elif rsi <= 60:
            return 50, "Neutral"
        elif rsi <= 70:
            return 35, "Approaching overbought"
        else:
            return 15, "Overbought"
    else:  # short
        if rsi >= 80:
            return 95, "Extreme overbought"
        elif rsi >= 70:
            return 85, "Overbought"
        elif rsi >= 60:
            return 70, "Approaching overbought"
        elif rsi >= 40:
            return 50, "Neutral"
        elif rsi >= 30:
            return 35, "Approaching oversold"
        else:
            return 15, "Oversold"


def volume_ratio_to_score(ratio: float) -> Tuple[float, str]:
    """
    Convert volume ratio to score.

    Ratio = current volume / average volume
    Higher ratio = more participation = stronger signal
    """
    if ratio >= 3.0:
        return 95, f"Volume spike {ratio:.1f}x"
    elif ratio >= 2.0:
        return 85, f"High volume {ratio:.1f}x"
    elif ratio >= 1.5:
        return 70, f"Above avg {ratio:.1f}x"
    elif ratio >= 0.8:
        return 50, f"Normal volume"
    elif ratio >= 0.5:
        return 30, f"Low volume"
    else:
        return 15, f"Very low volume"


def imbalance_to_score(imbalance: float, direction: str = "long") -> Tuple[float, str]:
    """
    Convert order book imbalance to score.

    Imbalance = bid_volume / (bid_volume + ask_volume)
    > 0.6 = buying pressure
    < 0.4 = selling pressure
    """
    if direction == "long":
        if imbalance >= 0.75:
            return 95, f"{imbalance*100:.0f}% bid imbalance"
        elif imbalance >= 0.6:
            return 80, f"{imbalance*100:.0f}% bid heavy"
        elif imbalance >= 0.45:
            return 50, "Balanced book"
        elif imbalance >= 0.35:
            return 30, f"{(1-imbalance)*100:.0f}% ask heavy"
        else:
            return 15, f"{(1-imbalance)*100:.0f}% ask imbalance"
    else:  # short
        if imbalance <= 0.25:
            return 95, f"{(1-imbalance)*100:.0f}% ask imbalance"
        elif imbalance <= 0.4:
            return 80, f"{(1-imbalance)*100:.0f}% ask heavy"
        elif imbalance <= 0.55:
            return 50, "Balanced book"
        elif imbalance <= 0.65:
            return 30, f"{imbalance*100:.0f}% bid heavy"
        else:
            return 15, f"{imbalance*100:.0f}% bid imbalance"


def funding_to_score(funding_rate: float, direction: str = "long") -> Tuple[float, str]:
    """
    Convert funding rate to score (contrarian indicator).

    Positive funding = longs pay shorts = crowded longs
    Negative funding = shorts pay longs = crowded shorts

    For longs: negative/low funding is bullish (less crowded)
    For shorts: positive/high funding is bullish (longs crowded)
    """
    rate_pct = funding_rate * 100

    if direction == "long":
        if rate_pct <= -0.05:
            return 90, "Shorts crowded (contrarian buy)"
        elif rate_pct <= -0.01:
            return 75, "Slightly short-biased"
        elif rate_pct <= 0.01:
            return 50, "Neutral funding"
        elif rate_pct <= 0.05:
            return 35, "Slightly long-biased"
        else:
            return 15, "Longs crowded (caution)"
    else:  # short
        if rate_pct >= 0.05:
            return 90, "Longs crowded (contrarian sell)"
        elif rate_pct >= 0.01:
            return 75, "Slightly long-biased"
        elif rate_pct >= -0.01:
            return 50, "Neutral funding"
        elif rate_pct >= -0.05:
            return 35, "Slightly short-biased"
        else:
            return 15, "Shorts crowded (caution)"


def fear_greed_to_score(fng: int, direction: str = "long") -> Tuple[float, str]:
    """
    Convert Fear & Greed Index to score (contrarian indicator).

    0-25 = Extreme Fear → Contrarian buy
    25-45 = Fear
    45-55 = Neutral
    55-75 = Greed
    75-100 = Extreme Greed → Contrarian sell
    """
    if direction == "long":
        if fng <= 20:
            return 95, f"Extreme Fear ({fng}) - contrarian buy"
        elif fng <= 35:
            return 80, f"Fear ({fng})"
        elif fng <= 55:
            return 50, f"Neutral ({fng})"
        elif fng <= 75:
            return 35, f"Greed ({fng})"
        else:
            return 15, f"Extreme Greed ({fng}) - caution"
    else:  # short
        if fng >= 80:
            return 95, f"Extreme Greed ({fng}) - contrarian sell"
        elif fng >= 65:
            return 80, f"Greed ({fng})"
        elif fng >= 45:
            return 50, f"Neutral ({fng})"
        elif fng >= 25:
            return 35, f"Fear ({fng})"
        else:
            return 15, f"Extreme Fear ({fng}) - caution"


def apply_freshness_decay(score: float, age_minutes: float) -> float:
    """
    Apply decay to score based on data age.

    < 5 min = full score
    5-15 min = 70% score
    15-30 min = 40% score
    > 30 min = 20% score
    """
    if age_minutes < 5:
        decay = 1.0
    elif age_minutes < 15:
        decay = 0.7
    elif age_minutes < 30:
        decay = 0.4
    else:
        decay = 0.2

    # Decay towards neutral (50)
    decayed = 50 + (score - 50) * decay
    return round(decayed, 1)


def combine_scores(scores: dict, weights: dict, default_weight: float = 0.1) -> float:
    """
    Combine multiple scores with weights.

    Args:
        scores: Dict of {indicator: score}
        weights: Dict of {indicator: weight}
        default_weight: Weight for indicators not in weights dict

    Returns:
        Weighted average score (0-100)
    """
    total_score = 0
    total_weight = 0

    for indicator, score in scores.items():
        weight = weights.get(indicator, default_weight)
        total_score += score * weight
        total_weight += weight

    if total_weight == 0:
        return 50.0

    return round(total_score / total_weight, 1)
