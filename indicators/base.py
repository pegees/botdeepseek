"""Base classes for all indicators."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class IndicatorResult:
    """Standard output format for all indicators."""
    name: str
    value: Any                          # Primary value (e.g., RSI = 75.5)
    signal: str                         # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float                     # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "signal": self.signal,
            "strength": self.strength,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return f"{self.name}: {self.signal} ({self.strength:.0%})"


class IndicatorBase(ABC):
    """Abstract base class for all indicators."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**self.default_params(), **(params or {})}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this indicator."""
        pass

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for this indicator."""
        pass

    @abstractmethod
    def calculate(self, candles: List[Dict]) -> IndicatorResult:
        """
        Calculate indicator from OHLCV candle data.

        Args:
            candles: List of dicts with keys: open, high, low, close, volume

        Returns:
            IndicatorResult with signal and metadata
        """
        pass

    def validate_candles(self, candles: List[Dict], min_candles: int = 20) -> bool:
        """Validate input candles have required fields and minimum length."""
        if len(candles) < min_candles:
            return False

        required = {"open", "high", "low", "close", "volume"}
        if candles:
            return required.issubset(candles[0].keys())
        return False

    def extract_series(self, candles: List[Dict], field: str) -> List[float]:
        """Extract a single series from candles."""
        return [float(c[field]) for c in candles]
