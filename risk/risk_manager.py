"""
Risk Manager - ATR-Based Position Sizing & Multi-Level TP
=========================================================
Calculates stop loss, take profits, and position sizes.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from config import RISK_CONFIG, DEFAULT_USER_SETTINGS

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class TakeProfit:
    """Take profit level."""
    level: int           # TP1, TP2, TP3
    price: float
    rr_ratio: float      # Reward:Risk ratio
    size_pct: float      # % of position to close
    distance_pct: float  # Distance from entry


@dataclass
class RiskCalculation:
    """Complete risk calculation for a trade."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    sl_distance_pct: float
    take_profits: List[TakeProfit]
    position_size_usd: float
    position_size_units: float
    risk_amount_usd: float
    risk_pct: float
    rr_ratio: float           # Overall R:R (to TP2)
    atr: float
    atr_multiplier: float

    def format_summary(self) -> str:
        """Format as trading summary."""
        emoji = "🟢" if self.direction == TradeDirection.LONG else "🔴"
        direction = "LONG" if self.direction == TradeDirection.LONG else "SHORT"

        tp_lines = []
        for tp in self.take_profits:
            action = "SL→BE" if tp.level == 1 else "Trail" if tp.level == 2 else ""
            tp_lines.append(
                f"├─ TP{tp.level}: ${tp.price:.4f} → {tp.size_pct*100:.0f}% | {action}"
            )

        tp_section = "\n".join(tp_lines)

        return f"""
{emoji} {direction} — {self.symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Entry: ${self.entry_price:.4f}
├─ SL: ${self.stop_loss:.4f} ({self.sl_distance_pct:.2f}%)
{tp_section}

⚖️ R:R 1:{self.rr_ratio:.1f} | Size: ${self.position_size_usd:.0f}
💵 Risk: ${self.risk_amount_usd:.0f} ({self.risk_pct:.1f}%)
"""


class RiskManager:
    """
    Manages risk calculations for trades.
    Uses ATR for stop loss, multi-level take profits.
    """

    def __init__(self, account_balance: float = None, risk_per_trade_pct: float = None):
        """
        Initialize risk manager.

        Args:
            account_balance: Trading account balance in USD
            risk_per_trade_pct: Risk per trade as percentage (e.g., 2.0 for 2%)
        """
        self.account_balance = account_balance or DEFAULT_USER_SETTINGS.account_balance
        self.risk_per_trade_pct = risk_per_trade_pct or RISK_CONFIG.risk_per_trade_pct

    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.account_balance = new_balance
        logger.info(f"Account balance updated to ${new_balance:.2f}")

    def update_risk_pct(self, new_risk_pct: float):
        """Update risk percentage."""
        self.risk_per_trade_pct = new_risk_pct
        logger.info(f"Risk per trade updated to {new_risk_pct}%")

    def calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = None
    ) -> float:
        """
        Calculate Average True Range.

        ATR = Average of True Range over period
        True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        """
        period = period or RISK_CONFIG.atr_period

        if len(highs) < period + 1:
            # Fallback: use simple high-low range
            return np.mean([h - l for h, l in zip(highs[-period:], lows[-period:])])

        true_ranges = []
        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # ATR is the average of last 'period' true ranges
        return np.mean(true_ranges[-period:])

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: TradeDirection,
        multiplier: float = None
    ) -> Tuple[float, float]:
        """
        Calculate stop loss based on ATR.

        Returns:
            (stop_loss_price, distance_percentage)
        """
        multiplier = multiplier or RISK_CONFIG.atr_sl_multiplier
        sl_distance = atr * multiplier

        if direction == TradeDirection.LONG:
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance

        distance_pct = (sl_distance / entry_price) * 100

        return stop_loss, distance_pct

    def calculate_take_profits(
        self,
        entry_price: float,
        stop_loss: float,
        direction: TradeDirection
    ) -> List[TakeProfit]:
        """
        Calculate multi-level take profits.

        TP1: 1.5R (40% of position) - Move SL to break-even
        TP2: 2.5R (40% of position) - Trail stop
        TP3: 4.0R (20% of position) - Runner
        """
        # Risk per unit
        if direction == TradeDirection.LONG:
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price

        take_profits = []

        # TP1
        tp1_distance = risk_per_unit * RISK_CONFIG.tp1_rr
        if direction == TradeDirection.LONG:
            tp1_price = entry_price + tp1_distance
        else:
            tp1_price = entry_price - tp1_distance

        take_profits.append(TakeProfit(
            level=1,
            price=tp1_price,
            rr_ratio=RISK_CONFIG.tp1_rr,
            size_pct=RISK_CONFIG.tp1_size,
            distance_pct=(tp1_distance / entry_price) * 100
        ))

        # TP2
        tp2_distance = risk_per_unit * RISK_CONFIG.tp2_rr
        if direction == TradeDirection.LONG:
            tp2_price = entry_price + tp2_distance
        else:
            tp2_price = entry_price - tp2_distance

        take_profits.append(TakeProfit(
            level=2,
            price=tp2_price,
            rr_ratio=RISK_CONFIG.tp2_rr,
            size_pct=RISK_CONFIG.tp2_size,
            distance_pct=(tp2_distance / entry_price) * 100
        ))

        # TP3
        tp3_distance = risk_per_unit * RISK_CONFIG.tp3_rr
        if direction == TradeDirection.LONG:
            tp3_price = entry_price + tp3_distance
        else:
            tp3_price = entry_price - tp3_distance

        take_profits.append(TakeProfit(
            level=3,
            price=tp3_price,
            rr_ratio=RISK_CONFIG.tp3_rr,
            size_pct=RISK_CONFIG.tp3_size,
            distance_pct=(tp3_distance / entry_price) * 100
        ))

        return take_profits

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk.

        Position Size = Risk Amount / (Entry - SL)

        Returns:
            (position_size_usd, position_size_units)
        """
        risk_amount = self.account_balance * (self.risk_per_trade_pct / 100)
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance == 0:
            logger.warning("SL distance is 0, using 1% as fallback")
            sl_distance = entry_price * 0.01

        # Position size in units (e.g., number of BTC)
        position_size_units = risk_amount / sl_distance

        # Position size in USD
        position_size_usd = position_size_units * entry_price

        return position_size_usd, position_size_units

    def calculate_full_risk(
        self,
        symbol: str,
        entry_price: float,
        direction: TradeDirection,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        atr_multiplier: float = None
    ) -> RiskCalculation:
        """
        Calculate complete risk parameters for a trade.

        This is the main function to call for trade setup.
        """
        # Calculate ATR
        atr = self.calculate_atr(highs, lows, closes)

        # Calculate stop loss
        stop_loss, sl_distance_pct = self.calculate_stop_loss(
            entry_price, atr, direction, atr_multiplier
        )

        # Calculate take profits
        take_profits = self.calculate_take_profits(entry_price, stop_loss, direction)

        # Calculate position size
        position_size_usd, position_size_units = self.calculate_position_size(
            entry_price, stop_loss
        )

        # Risk amount
        risk_amount = self.account_balance * (self.risk_per_trade_pct / 100)

        # Overall R:R (to TP2)
        rr_ratio = RISK_CONFIG.tp2_rr

        return RiskCalculation(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            sl_distance_pct=sl_distance_pct,
            take_profits=take_profits,
            position_size_usd=position_size_usd,
            position_size_units=position_size_units,
            risk_amount_usd=risk_amount,
            risk_pct=self.risk_per_trade_pct,
            rr_ratio=rr_ratio,
            atr=atr,
            atr_multiplier=atr_multiplier or RISK_CONFIG.atr_sl_multiplier
        )

    def validate_risk(self, calculation: RiskCalculation) -> Tuple[bool, str]:
        """
        Validate that risk parameters are acceptable.

        Returns:
            (is_valid, reason)
        """
        # Check minimum R:R
        if calculation.rr_ratio < RISK_CONFIG.min_rr_ratio:
            return False, f"R:R {calculation.rr_ratio:.1f} below minimum {RISK_CONFIG.min_rr_ratio}"

        # Check position size vs max heat
        position_risk_pct = (calculation.risk_amount_usd / self.account_balance) * 100
        if position_risk_pct > RISK_CONFIG.max_portfolio_heat:
            return False, f"Position risk {position_risk_pct:.1f}% exceeds max heat {RISK_CONFIG.max_portfolio_heat}%"

        # Check SL distance (shouldn't be too tight or too wide)
        if calculation.sl_distance_pct < 0.1:
            return False, f"SL too tight ({calculation.sl_distance_pct:.2f}%)"
        if calculation.sl_distance_pct > 10:
            return False, f"SL too wide ({calculation.sl_distance_pct:.2f}%)"

        return True, "OK"

    def get_quick_levels(
        self,
        entry_price: float,
        sl_pct: float,
        direction: TradeDirection
    ) -> Dict[str, float]:
        """
        Quick calculation with percentage-based SL.
        For when ATR data isn't available.
        """
        if direction == TradeDirection.LONG:
            stop_loss = entry_price * (1 - sl_pct / 100)
        else:
            stop_loss = entry_price * (1 + sl_pct / 100)

        tps = self.calculate_take_profits(entry_price, stop_loss, direction)

        return {
            "entry": entry_price,
            "sl": stop_loss,
            "tp1": tps[0].price,
            "tp2": tps[1].price,
            "tp3": tps[2].price,
        }


# Singleton
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get the risk manager singleton."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


def calculate_trade_risk(
    symbol: str,
    entry_price: float,
    direction: TradeDirection,
    highs: List[float],
    lows: List[float],
    closes: List[float]
) -> RiskCalculation:
    """Convenience function for calculating trade risk."""
    rm = get_risk_manager()
    return rm.calculate_full_risk(symbol, entry_price, direction, highs, lows, closes)
