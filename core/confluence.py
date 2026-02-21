"""
Confluence Scoring Engine - 4-Layer Brain
==========================================
Combines Technical, Order Flow, On-Chain, and Sentiment data
into a single confidence score (0-100).
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config import LAYER_WEIGHTS, SIGNAL_THRESHOLDS
from data_layers.technical import TechnicalScore, SignalDirection

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "strong"      # Score >= 80
    MEDIUM = "medium"      # Score 65-79
    WEAK = "weak"          # Score 50-64
    SKIP = "skip"          # Score < 50


@dataclass
class LayerScore:
    """Score from a single data layer."""
    name: str
    score: float              # 0-100
    weight: float             # Weight in confluence
    weighted_score: float     # score * weight
    direction: SignalDirection
    summary: str
    details: Dict = None


@dataclass
class ConfluenceResult:
    """Result of confluence analysis."""
    total_score: float        # 0-100 weighted
    direction: SignalDirection
    strength: SignalStrength
    layer_scores: List[LayerScore]

    # Individual layer results
    ta_score: float
    ta_summary: str
    orderflow_score: float
    orderflow_summary: str
    onchain_score: float
    onchain_summary: str
    sentiment_score: float
    sentiment_summary: str
    backtest_score: float
    backtest_summary: str

    # Meta
    timestamp: datetime
    symbol: str
    timeframe: str

    @property
    def is_tradeable(self) -> bool:
        """Check if score meets minimum threshold."""
        return self.total_score >= SIGNAL_THRESHOLDS.minimum_score

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence signal."""
        return self.total_score >= SIGNAL_THRESHOLDS.high_confidence

    def get_summary(self) -> str:
        """Get a text summary of the confluence."""
        stars = "⭐" * min(5, int(self.total_score / 20))

        lines = [
            f"📊 Confluence: {self.total_score:.0f}/100 {stars}",
            f"Direction: {self.direction.value.upper()}",
            f"Strength: {self.strength.value.upper()}",
            "",
            "Layer Breakdown:",
        ]

        for layer in self.layer_scores:
            lines.append(f"  {layer.name}: {layer.score:.0f} ({layer.summary})")

        return "\n".join(lines)


class ConfluenceEngine:
    """
    The brain of the trading system.
    Combines 4 data layers into one confluence score.

    Weights:
    - Technical: 35%
    - Order Flow: 25%
    - On-Chain: 20%
    - Sentiment: 10%
    - Backtest: 10%
    """

    def __init__(self):
        self.weights = LAYER_WEIGHTS

    def calculate_confluence(
        self,
        symbol: str,
        timeframe: str,
        technical: Optional[TechnicalScore] = None,
        orderflow_score: float = 50,
        orderflow_summary: str = "No data",
        onchain_score: float = 50,
        onchain_summary: str = "No data",
        sentiment_score: float = 50,
        sentiment_summary: str = "No data",
        backtest_score: float = 50,
        backtest_summary: str = "No data"
    ) -> ConfluenceResult:
        """
        Calculate confluence score from all data layers.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Analysis timeframe (e.g., "15m")
            technical: TechnicalScore from TA analysis
            orderflow_score: Order flow score (0-100)
            orderflow_summary: Order flow summary text
            onchain_score: On-chain score (0-100)
            onchain_summary: On-chain summary text
            sentiment_score: Sentiment score (0-100)
            sentiment_summary: Sentiment summary text
            backtest_score: Backtest score (0-100)
            backtest_summary: Backtest summary text

        Returns:
            ConfluenceResult with weighted score and breakdown
        """
        layer_scores: List[LayerScore] = []

        # 1. Technical Analysis (35%)
        if technical:
            ta_score = technical.total_score
            ta_summary = technical.summary
            ta_direction = technical.direction
        else:
            ta_score = 50
            ta_summary = "No TA data"
            ta_direction = SignalDirection.NEUTRAL

        layer_scores.append(LayerScore(
            name="Technical",
            score=ta_score,
            weight=self.weights.technical,
            weighted_score=ta_score * self.weights.technical,
            direction=ta_direction,
            summary=ta_summary
        ))

        # 2. Order Flow (25%)
        # Convert score to direction
        if orderflow_score >= 65:
            of_direction = SignalDirection.LONG
        elif orderflow_score <= 35:
            of_direction = SignalDirection.SHORT
        else:
            of_direction = SignalDirection.NEUTRAL

        layer_scores.append(LayerScore(
            name="OrderFlow",
            score=orderflow_score,
            weight=self.weights.orderflow,
            weighted_score=orderflow_score * self.weights.orderflow,
            direction=of_direction,
            summary=orderflow_summary
        ))

        # 3. On-Chain (20%)
        if onchain_score >= 65:
            oc_direction = SignalDirection.LONG
        elif onchain_score <= 35:
            oc_direction = SignalDirection.SHORT
        else:
            oc_direction = SignalDirection.NEUTRAL

        layer_scores.append(LayerScore(
            name="OnChain",
            score=onchain_score,
            weight=self.weights.onchain,
            weighted_score=onchain_score * self.weights.onchain,
            direction=oc_direction,
            summary=onchain_summary
        ))

        # 4. Sentiment (10%)
        if sentiment_score >= 65:
            sent_direction = SignalDirection.LONG
        elif sentiment_score <= 35:
            sent_direction = SignalDirection.SHORT
        else:
            sent_direction = SignalDirection.NEUTRAL

        layer_scores.append(LayerScore(
            name="Sentiment",
            score=sentiment_score,
            weight=self.weights.sentiment,
            weighted_score=sentiment_score * self.weights.sentiment,
            direction=sent_direction,
            summary=sentiment_summary
        ))

        # 5. Backtest (10%)
        if backtest_score >= 65:
            bt_direction = SignalDirection.LONG
        elif backtest_score <= 35:
            bt_direction = SignalDirection.SHORT
        else:
            bt_direction = SignalDirection.NEUTRAL

        layer_scores.append(LayerScore(
            name="Backtest",
            score=backtest_score,
            weight=self.weights.backtest,
            weighted_score=backtest_score * self.weights.backtest,
            direction=bt_direction,
            summary=backtest_summary
        ))

        # Calculate total weighted score
        total_score = sum(layer.weighted_score for layer in layer_scores)

        # Determine overall direction (majority vote weighted by score)
        long_weight = sum(
            layer.weighted_score
            for layer in layer_scores
            if layer.direction == SignalDirection.LONG
        )
        short_weight = sum(
            layer.weighted_score
            for layer in layer_scores
            if layer.direction == SignalDirection.SHORT
        )

        if long_weight > short_weight and total_score >= 55:
            direction = SignalDirection.LONG
        elif short_weight > long_weight and total_score <= 45:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Determine signal strength
        if total_score >= SIGNAL_THRESHOLDS.high_confidence:
            strength = SignalStrength.STRONG
        elif total_score >= SIGNAL_THRESHOLDS.medium_confidence:
            strength = SignalStrength.MEDIUM
        elif total_score >= SIGNAL_THRESHOLDS.low_confidence:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.SKIP

        return ConfluenceResult(
            total_score=round(total_score, 1),
            direction=direction,
            strength=strength,
            layer_scores=layer_scores,
            ta_score=ta_score,
            ta_summary=ta_summary,
            orderflow_score=orderflow_score,
            orderflow_summary=orderflow_summary,
            onchain_score=onchain_score,
            onchain_summary=onchain_summary,
            sentiment_score=sentiment_score,
            sentiment_summary=sentiment_summary,
            backtest_score=backtest_score,
            backtest_summary=backtest_summary,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe
        )

    def quick_confluence(
        self,
        symbol: str,
        ta_score: float,
        ta_summary: str = "",
        ta_direction: SignalDirection = SignalDirection.NEUTRAL
    ) -> ConfluenceResult:
        """
        Quick confluence with just TA data.
        Other layers default to neutral (50).
        """
        technical = TechnicalScore(
            total_score=ta_score,
            direction=ta_direction,
            indicator_scores=[],
            summary=ta_summary
        )

        return self.calculate_confluence(
            symbol=symbol,
            timeframe="15m",
            technical=technical
        )

    def should_signal(
        self,
        result: ConfluenceResult,
        threshold_adjustment: float = 0
    ) -> Tuple[bool, str]:
        """
        Determine if a signal should be sent.

        Args:
            result: ConfluenceResult from calculate_confluence
            threshold_adjustment: Additional threshold boost (from circuit breakers)

        Returns:
            (should_send, reason)
        """
        effective_threshold = SIGNAL_THRESHOLDS.minimum_score + threshold_adjustment

        # Check score threshold
        if result.total_score < effective_threshold:
            return False, f"Score {result.total_score:.0f} below threshold {effective_threshold:.0f}"

        # Check direction
        if result.direction == SignalDirection.NEUTRAL:
            return False, "Direction is neutral"

        # Check if all layers agree
        disagreeing_layers = [
            layer.name for layer in result.layer_scores
            if layer.direction != SignalDirection.NEUTRAL
            and layer.direction != result.direction
        ]

        if len(disagreeing_layers) >= 2:
            return False, f"Layers disagree: {', '.join(disagreeing_layers)}"

        return True, "Signal approved"


# Singleton
_engine: Optional[ConfluenceEngine] = None


def get_confluence_engine() -> ConfluenceEngine:
    """Get the confluence engine singleton."""
    global _engine
    if _engine is None:
        _engine = ConfluenceEngine()
    return _engine


def calculate_confluence(
    symbol: str,
    technical: Optional[TechnicalScore] = None,
    **kwargs
) -> ConfluenceResult:
    """Convenience function for calculating confluence."""
    engine = get_confluence_engine()
    return engine.calculate_confluence(
        symbol=symbol,
        timeframe="15m",
        technical=technical,
        **kwargs
    )
