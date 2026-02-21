"""Technical indicators for market analysis."""
from .base import IndicatorBase, IndicatorResult
from .registry import IndicatorRegistry

# Import all indicator modules to trigger auto-registration
from .momentum import RSIIndicator, MACDIndicator
from .trend import EMAIndicator
from .volume import VolumeIndicator, CVDIndicator
from .structure import MarketStructureIndicator
from .levels import SupportResistanceIndicator, LiquiditySweepIndicator, FVGIndicator
from .derivatives import WhaleIndicator, OpenInterestIndicator

__all__ = [
    # Base classes
    "IndicatorBase",
    "IndicatorResult",
    "IndicatorRegistry",
    # Momentum indicators
    "RSIIndicator",
    "MACDIndicator",
    # Trend indicators
    "EMAIndicator",
    # Volume indicators
    "VolumeIndicator",
    "CVDIndicator",
    # Structure indicators
    "MarketStructureIndicator",
    # Level indicators
    "SupportResistanceIndicator",
    "LiquiditySweepIndicator",
    "FVGIndicator",
    # Derivatives indicators
    "WhaleIndicator",
    "OpenInterestIndicator",
]
