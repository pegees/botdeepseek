"""
Pairs Configuration
====================
Pair filtering thresholds and exclusion lists.
"""
from typing import List, Set

# Minimum volume thresholds
MIN_24H_VOLUME_USDT = 500_000          # $500K minimum
MIN_LIQUIDITY_SCORE = 60                # 0-100 scale
MAX_SPREAD_PCT = 0.10                   # Maximum 0.1% spread

# Excluded pairs (stablecoins, leveraged tokens, etc.)
EXCLUDED_PAIRS: Set[str] = {
    # Stablecoins
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "EURUSDT",
    "USDPUSDT", "FDUSDUSDT", "USTCUSDT",

    # Leveraged tokens
    "BTCUPUSDT", "BTCDOWNUSDT",
    "ETHUPUSDT", "ETHDOWNUSDT",
    "BNBUPUSDT", "BNBDOWNUSDT",
}

# Preferred pairs (get priority in scanning)
PREFERRED_PAIRS: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
    "DOTUSDT", "MATICUSDT", "LINKUSDT", "ATOMUSDT",
]

# Correlation groups (pairs that move together)
CORRELATION_GROUPS = {
    "eth_ecosystem": ["ETHUSDT", "MATICUSDT", "ARBUSDT", "OPUSDT", "STXUSDT"],
    "layer1": ["SOLUSDT", "AVAXUSDT", "ATOMUSDT", "DOTUSDT", "NEARUSDT"],
    "meme": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT"],
    "defi": ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT"],
}

# Asian trading hours pairs (higher volume during Asia session)
ASIA_FOCUS_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]

# US trading hours pairs (higher volume during US session)
US_FOCUS_PAIRS = ["SOLUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"]


def is_excluded(symbol: str) -> bool:
    """Check if a symbol should be excluded."""
    # Direct exclusion
    if symbol in EXCLUDED_PAIRS:
        return True

    # Pattern-based exclusion
    exclude_patterns = ["UP", "DOWN", "BULL", "BEAR", "3L", "3S"]
    for pattern in exclude_patterns:
        if pattern in symbol.upper():
            return True

    return False


def get_correlation_group(symbol: str) -> str:
    """Get the correlation group for a symbol."""
    for group_name, pairs in CORRELATION_GROUPS.items():
        if symbol in pairs:
            return group_name
    return "independent"


def is_preferred(symbol: str) -> bool:
    """Check if symbol is in preferred list."""
    return symbol in PREFERRED_PAIRS
