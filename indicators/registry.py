"""Indicator registry for discovering and instantiating indicators."""
from typing import Dict, Type, List, Optional
import logging

from .base import IndicatorBase, IndicatorResult

logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """Registry for discovering and instantiating indicators."""

    _indicators: Dict[str, Type[IndicatorBase]] = {}

    @classmethod
    def register(cls, indicator_class: Type[IndicatorBase]) -> Type[IndicatorBase]:
        """Decorator to register an indicator."""
        instance = indicator_class()
        cls._indicators[instance.name] = indicator_class
        logger.debug(f"Registered indicator: {instance.name}")
        return indicator_class

    @classmethod
    def get(cls, name: str, params: Optional[Dict] = None) -> IndicatorBase:
        """Get an indicator instance by name."""
        if name not in cls._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        return cls._indicators[name](params)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered indicator names."""
        return list(cls._indicators.keys())

    @classmethod
    def calculate_all(
        cls,
        candles: List[Dict],
        indicator_names: Optional[List[str]] = None,
        params: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, IndicatorResult]:
        """
        Calculate multiple indicators at once.

        Args:
            candles: OHLCV candle data
            indicator_names: List of indicator names (None = all)
            params: Dict of indicator_name -> params dict

        Returns:
            Dict of indicator_name -> IndicatorResult
        """
        names = indicator_names or cls.list_all()
        params = params or {}
        results = {}

        for name in names:
            try:
                indicator = cls.get(name, params.get(name))
                if indicator.validate_candles(candles):
                    results[name] = indicator.calculate(candles)
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                continue

        return results

    @classmethod
    def get_confluence(cls, results: Dict[str, IndicatorResult]) -> Dict[str, int]:
        """
        Count bullish/bearish/neutral signals.

        Returns:
            Dict with counts: {"bullish": N, "bearish": N, "neutral": N}
        """
        counts = {"bullish": 0, "bearish": 0, "neutral": 0}

        for result in results.values():
            if result.signal == "BULLISH":
                counts["bullish"] += 1
            elif result.signal == "BEARISH":
                counts["bearish"] += 1
            else:
                counts["neutral"] += 1

        return counts
