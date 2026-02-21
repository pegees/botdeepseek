"""
Circuit Breakers - 9 Anti-Emotion Safeguards
=============================================
Prevents revenge trading, overtrading, and emotional decisions.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from config import CIRCUIT_BREAKERS

logger = logging.getLogger(__name__)


class BreakerType(Enum):
    """Types of circuit breakers."""
    REVENGE_TRADE = "revenge_trade"
    OVERTRADING = "overtrading"
    DAILY_DRAWDOWN = "daily_drawdown"
    WIN_STREAK = "win_streak"
    CORRELATION = "correlation"
    TIME_FILTER = "time_filter"
    MAX_POSITIONS = "max_positions"
    VOLATILITY_EXTREME = "volatility_extreme"
    MACRO_DANGER = "macro_danger"


@dataclass
class BreakerStatus:
    """Status of a circuit breaker."""
    breaker_type: BreakerType
    is_triggered: bool
    reason: str
    cooldown_until: Optional[datetime] = None
    threshold_adjustment: float = 0  # For score threshold boost


@dataclass
class TradeRecord:
    """Record of a trade for tracking."""
    symbol: str
    direction: str  # "long" or "short"
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0
    exit_price: float = 0
    pnl_pct: float = 0
    is_win: bool = False
    is_loss: bool = False


@dataclass
class TradingSession:
    """Trading session state."""
    start_time: datetime = field(default_factory=datetime.now)
    signals_this_hour: int = 0
    hour_start: datetime = field(default_factory=datetime.now)
    daily_pnl_pct: float = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    last_loss_time: Optional[datetime] = None
    open_positions: List[str] = field(default_factory=list)
    recent_trades: List[TradeRecord] = field(default_factory=list)
    btc_1h_change: float = 0
    paused_until: Optional[datetime] = None


class CircuitBreakerManager:
    """
    Manages all 9 circuit breakers to prevent emotional trading.

    Circuit Breakers:
    1. Revenge trade prevention (cooldown after SL)
    2. Overtrading prevention (max signals/hour)
    3. Drawdown protection (pause at daily loss limit)
    4. Win streak sizing (reduce size after streak)
    5. Correlation filter (limit correlated positions)
    6. Time quality filter (boost threshold during bad hours)
    7. Max positions (limit concurrent trades)
    8. Volatility extreme (skip during extreme volatility)
    9. Macro danger (pause during BTC crash)
    """

    def __init__(self):
        self.session = TradingSession()
        self._position_correlations: Dict[str, Dict[str, float]] = {}

    def reset_session(self):
        """Reset trading session (call at start of day)."""
        self.session = TradingSession()
        logger.info("Trading session reset")

    def record_signal(self):
        """Record a signal being sent."""
        now = datetime.now()

        # Reset hourly counter if new hour
        if now.hour != self.session.hour_start.hour:
            self.session.signals_this_hour = 0
            self.session.hour_start = now

        self.session.signals_this_hour += 1

    def record_trade_result(self, trade: TradeRecord):
        """Record a completed trade."""
        self.session.recent_trades.append(trade)

        # Keep only last 100 trades
        if len(self.session.recent_trades) > 100:
            self.session.recent_trades = self.session.recent_trades[-100:]

        # Update session state
        self.session.daily_pnl_pct += trade.pnl_pct

        if trade.is_win:
            self.session.consecutive_wins += 1
            self.session.consecutive_losses = 0
        elif trade.is_loss:
            self.session.consecutive_losses += 1
            self.session.consecutive_wins = 0
            self.session.last_loss_time = trade.exit_time

        # Remove from open positions
        if trade.symbol in self.session.open_positions:
            self.session.open_positions.remove(trade.symbol)

    def add_position(self, symbol: str):
        """Add a new open position."""
        if symbol not in self.session.open_positions:
            self.session.open_positions.append(symbol)

    def update_btc_change(self, pct_change_1h: float):
        """Update BTC 1h price change."""
        self.session.btc_1h_change = pct_change_1h

    def set_correlation(self, symbol1: str, symbol2: str, correlation: float):
        """Set correlation between two symbols."""
        if symbol1 not in self._position_correlations:
            self._position_correlations[symbol1] = {}
        self._position_correlations[symbol1][symbol2] = correlation

    # =========================================================================
    # Individual Circuit Breakers
    # =========================================================================

    def check_revenge_trade(self) -> BreakerStatus:
        """
        Breaker 1: Revenge Trade Prevention
        Cooldown period after hitting stop loss.
        """
        if self.session.last_loss_time is None:
            return BreakerStatus(
                BreakerType.REVENGE_TRADE, False, "No recent losses"
            )

        cooldown = timedelta(seconds=CIRCUIT_BREAKERS.revenge_cooldown_seconds)
        cooldown_end = self.session.last_loss_time + cooldown
        now = datetime.now()

        if now < cooldown_end:
            return BreakerStatus(
                BreakerType.REVENGE_TRADE,
                True,
                f"Revenge cooldown: {(cooldown_end - now).seconds}s remaining",
                cooldown_until=cooldown_end
            )

        return BreakerStatus(
            BreakerType.REVENGE_TRADE, False, "Cooldown complete"
        )

    def check_overtrading(self) -> BreakerStatus:
        """
        Breaker 2: Overtrading Prevention
        Maximum signals per hour limit.
        """
        max_signals = CIRCUIT_BREAKERS.max_signals_per_hour

        if self.session.signals_this_hour >= max_signals:
            return BreakerStatus(
                BreakerType.OVERTRADING,
                True,
                f"Max signals reached: {self.session.signals_this_hour}/{max_signals}"
            )

        return BreakerStatus(
            BreakerType.OVERTRADING,
            False,
            f"Signals: {self.session.signals_this_hour}/{max_signals}"
        )

    def check_daily_drawdown(self) -> BreakerStatus:
        """
        Breaker 3: Drawdown Protection
        Pause trading if daily loss exceeds limit.
        """
        limit = CIRCUIT_BREAKERS.daily_drawdown_limit_pct

        if self.session.daily_pnl_pct <= -limit:
            return BreakerStatus(
                BreakerType.DAILY_DRAWDOWN,
                True,
                f"Daily drawdown limit hit: {self.session.daily_pnl_pct:.1f}% (limit: -{limit}%)"
            )

        return BreakerStatus(
            BreakerType.DAILY_DRAWDOWN,
            False,
            f"Daily P&L: {self.session.daily_pnl_pct:+.1f}%"
        )

    def check_win_streak(self) -> BreakerStatus:
        """
        Breaker 4: Win Streak Sizing
        Reduce position size after consecutive wins (reversion to mean).
        """
        streak = self.session.consecutive_wins
        trigger = CIRCUIT_BREAKERS.win_streak_reduce_after

        if streak >= trigger:
            reduction = CIRCUIT_BREAKERS.win_streak_size_reduction
            return BreakerStatus(
                BreakerType.WIN_STREAK,
                True,
                f"Win streak ({streak}): Reduce size by {reduction*100:.0f}%"
            )

        return BreakerStatus(
            BreakerType.WIN_STREAK,
            False,
            f"Win streak: {streak}"
        )

    def check_correlation(self, new_symbol: str) -> BreakerStatus:
        """
        Breaker 5: Correlation Filter
        Limit highly correlated positions.
        """
        max_corr = CIRCUIT_BREAKERS.max_correlated_positions
        threshold = RISK_CONFIG.max_correlation if hasattr(RISK_CONFIG, 'max_correlation') else 0.85

        correlated_count = 0
        correlated_symbols = []

        for open_symbol in self.session.open_positions:
            # Check correlation
            corr = self._position_correlations.get(new_symbol, {}).get(open_symbol, 0)
            if abs(corr) >= threshold:
                correlated_count += 1
                correlated_symbols.append(open_symbol)

        if correlated_count >= max_corr:
            return BreakerStatus(
                BreakerType.CORRELATION,
                True,
                f"Too many correlated positions: {correlated_symbols}"
            )

        return BreakerStatus(
            BreakerType.CORRELATION,
            False,
            f"Correlated positions: {correlated_count}/{max_corr}"
        )

    def check_time_filter(self) -> BreakerStatus:
        """
        Breaker 6: Time Quality Filter
        Boost score threshold during low-quality hours.
        """
        current_hour = datetime.now().hour
        low_quality_hours = CIRCUIT_BREAKERS.low_quality_hours
        boost = CIRCUIT_BREAKERS.low_quality_threshold_boost

        if current_hour in low_quality_hours:
            return BreakerStatus(
                BreakerType.TIME_FILTER,
                True,
                f"Low quality hour ({current_hour:02d}:00): +{boost} threshold",
                threshold_adjustment=boost
            )

        return BreakerStatus(
            BreakerType.TIME_FILTER,
            False,
            f"Hour {current_hour:02d}:00 OK"
        )

    def check_max_positions(self) -> BreakerStatus:
        """
        Breaker 7: Max Positions
        Limit concurrent open positions.
        """
        max_pos = CIRCUIT_BREAKERS.max_open_positions
        current = len(self.session.open_positions)

        if current >= max_pos:
            return BreakerStatus(
                BreakerType.MAX_POSITIONS,
                True,
                f"Max positions reached: {current}/{max_pos}"
            )

        return BreakerStatus(
            BreakerType.MAX_POSITIONS,
            False,
            f"Open positions: {current}/{max_pos}"
        )

    def check_volatility_extreme(self, current_atr: float, normal_atr: float) -> BreakerStatus:
        """
        Breaker 8: Volatility Extreme
        Skip signals when volatility is abnormally high.
        """
        multiplier = CIRCUIT_BREAKERS.volatility_atr_multiplier

        if normal_atr > 0:
            ratio = current_atr / normal_atr
        else:
            ratio = 1.0

        if ratio >= multiplier:
            return BreakerStatus(
                BreakerType.VOLATILITY_EXTREME,
                True,
                f"Extreme volatility: {ratio:.1f}x normal ATR"
            )

        return BreakerStatus(
            BreakerType.VOLATILITY_EXTREME,
            False,
            f"Volatility: {ratio:.1f}x normal"
        )

    def check_macro_danger(self) -> BreakerStatus:
        """
        Breaker 9: Macro Danger (BTC Crash Protection)
        Pause if BTC drops significantly.
        """
        threshold = CIRCUIT_BREAKERS.btc_crash_threshold_pct
        btc_change = self.session.btc_1h_change

        if btc_change <= threshold:
            pause_minutes = CIRCUIT_BREAKERS.btc_crash_pause_minutes
            pause_until = datetime.now() + timedelta(minutes=pause_minutes)

            return BreakerStatus(
                BreakerType.MACRO_DANGER,
                True,
                f"BTC crash: {btc_change:.1f}% (1h). Pause {pause_minutes}min",
                cooldown_until=pause_until
            )

        return BreakerStatus(
            BreakerType.MACRO_DANGER,
            False,
            f"BTC 1h: {btc_change:+.1f}%"
        )

    # =========================================================================
    # Main Check Function
    # =========================================================================

    def check_all(
        self,
        new_symbol: str = None,
        current_atr: float = None,
        normal_atr: float = None
    ) -> Tuple[bool, List[BreakerStatus], float]:
        """
        Check all circuit breakers.

        Returns:
            (can_trade, breaker_statuses, threshold_adjustment)
        """
        statuses = []

        # 1. Revenge trade
        statuses.append(self.check_revenge_trade())

        # 2. Overtrading
        statuses.append(self.check_overtrading())

        # 3. Daily drawdown
        statuses.append(self.check_daily_drawdown())

        # 4. Win streak (warning, not blocking)
        statuses.append(self.check_win_streak())

        # 5. Correlation
        if new_symbol:
            statuses.append(self.check_correlation(new_symbol))

        # 6. Time filter
        statuses.append(self.check_time_filter())

        # 7. Max positions
        statuses.append(self.check_max_positions())

        # 8. Volatility extreme
        if current_atr is not None and normal_atr is not None:
            statuses.append(self.check_volatility_extreme(current_atr, normal_atr))

        # 9. Macro danger
        statuses.append(self.check_macro_danger())

        # Determine if trading is blocked
        blocking_breakers = [
            BreakerType.REVENGE_TRADE,
            BreakerType.OVERTRADING,
            BreakerType.DAILY_DRAWDOWN,
            BreakerType.MAX_POSITIONS,
            BreakerType.VOLATILITY_EXTREME,
            BreakerType.MACRO_DANGER,
        ]

        can_trade = True
        for status in statuses:
            if status.is_triggered and status.breaker_type in blocking_breakers:
                can_trade = False
                break

        # Calculate threshold adjustment
        threshold_adjustment = sum(
            s.threshold_adjustment for s in statuses if s.is_triggered
        )

        return can_trade, statuses, threshold_adjustment

    def get_status_summary(self) -> str:
        """Get a summary of all breaker statuses."""
        can_trade, statuses, adjustment = self.check_all()

        lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━"]
        lines.append("🔒 CIRCUIT BREAKERS")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

        for status in statuses:
            icon = "🔴" if status.is_triggered else "🟢"
            name = status.breaker_type.value.replace("_", " ").title()
            lines.append(f"{icon} {name}: {status.reason}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
        trade_status = "✅ TRADING ALLOWED" if can_trade else "❌ TRADING BLOCKED"
        lines.append(trade_status)

        if adjustment > 0:
            lines.append(f"⚠️ Score threshold +{adjustment}")

        return "\n".join(lines)


# Singleton
_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breakers() -> CircuitBreakerManager:
    """Get the circuit breaker manager singleton."""
    global _breaker_manager
    if _breaker_manager is None:
        _breaker_manager = CircuitBreakerManager()
    return _breaker_manager


# Import at end to avoid circular imports
try:
    from config import RISK_CONFIG
except ImportError:
    pass
