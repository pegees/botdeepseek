"""Cooldown management to prevent duplicate signals."""
import time
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CooldownEntry:
    """Represents a cooldown entry for a symbol."""
    symbol: str
    signal_type: str  # "BULLISH" or "BEARISH"
    timestamp: float
    expires_at: float


class CooldownManager:
    """
    Manages signal cooldowns to prevent spam.

    Features:
    - Per-symbol cooldowns (e.g., 60 min between signals for same symbol)
    - Per-direction tracking (can signal bullish after bearish)
    - Global rate limiting (max alerts per hour)
    """

    def __init__(
        self,
        cooldown_minutes: int = 60,
        max_alerts_per_hour: int = 10,
    ):
        """
        Initialize the cooldown manager.

        Args:
            cooldown_minutes: Minutes to wait before same signal on same symbol
            max_alerts_per_hour: Maximum alerts allowed per hour globally
        """
        self.cooldown_seconds = cooldown_minutes * 60
        self.max_alerts_per_hour = max_alerts_per_hour

        # Symbol -> CooldownEntry
        self._cooldowns: Dict[str, CooldownEntry] = {}

        # List of (timestamp, symbol) for global rate limiting
        self._alert_history: list = []

    def can_alert(self, symbol: str, signal_type: str) -> bool:
        """
        Check if an alert can be sent for this symbol/signal.

        Args:
            symbol: Trading pair symbol
            signal_type: "BULLISH" or "BEARISH"

        Returns:
            True if alert can be sent, False if on cooldown
        """
        now = time.time()

        # Clean up expired cooldowns
        self._cleanup_expired(now)

        # Check global rate limit
        if not self._check_global_rate_limit(now):
            logger.debug(f"Global rate limit reached, blocking {symbol}")
            return False

        # Check symbol-specific cooldown
        key = f"{symbol}:{signal_type}"
        if key in self._cooldowns:
            entry = self._cooldowns[key]
            if now < entry.expires_at:
                remaining = int(entry.expires_at - now)
                logger.debug(f"{symbol} on cooldown for {remaining}s more")
                return False

        return True

    def record_alert(self, symbol: str, signal_type: str) -> None:
        """
        Record that an alert was sent.

        Args:
            symbol: Trading pair symbol
            signal_type: "BULLISH" or "BEARISH"
        """
        now = time.time()

        # Record in symbol cooldown
        key = f"{symbol}:{signal_type}"
        self._cooldowns[key] = CooldownEntry(
            symbol=symbol,
            signal_type=signal_type,
            timestamp=now,
            expires_at=now + self.cooldown_seconds,
        )

        # Record in global history
        self._alert_history.append((now, symbol))

        logger.debug(f"Recorded alert for {symbol} ({signal_type})")

    def get_remaining_cooldown(self, symbol: str, signal_type: str) -> Optional[int]:
        """
        Get remaining cooldown time in seconds.

        Args:
            symbol: Trading pair symbol
            signal_type: "BULLISH" or "BEARISH"

        Returns:
            Seconds remaining, or None if not on cooldown
        """
        now = time.time()
        key = f"{symbol}:{signal_type}"

        if key in self._cooldowns:
            entry = self._cooldowns[key]
            if now < entry.expires_at:
                return int(entry.expires_at - now)

        return None

    def get_alerts_in_last_hour(self) -> int:
        """Get count of alerts sent in the last hour."""
        now = time.time()
        one_hour_ago = now - 3600

        return sum(1 for ts, _ in self._alert_history if ts > one_hour_ago)

    def _check_global_rate_limit(self, now: float) -> bool:
        """Check if global rate limit allows more alerts."""
        one_hour_ago = now - 3600
        recent_count = sum(1 for ts, _ in self._alert_history if ts > one_hour_ago)
        return recent_count < self.max_alerts_per_hour

    def _cleanup_expired(self, now: float) -> None:
        """Remove expired cooldown entries."""
        # Clean cooldowns
        expired_keys = [
            key for key, entry in self._cooldowns.items()
            if now >= entry.expires_at
        ]
        for key in expired_keys:
            del self._cooldowns[key]

        # Clean alert history (keep last hour only)
        one_hour_ago = now - 3600
        self._alert_history = [
            (ts, sym) for ts, sym in self._alert_history
            if ts > one_hour_ago
        ]

    def clear_all(self) -> None:
        """Clear all cooldowns (useful for testing)."""
        self._cooldowns.clear()
        self._alert_history.clear()

    def get_active_cooldowns(self) -> Dict[str, int]:
        """
        Get all active cooldowns with remaining time.

        Returns:
            Dict mapping symbol:signal_type to remaining seconds
        """
        now = time.time()
        self._cleanup_expired(now)

        return {
            key: int(entry.expires_at - now)
            for key, entry in self._cooldowns.items()
        }
