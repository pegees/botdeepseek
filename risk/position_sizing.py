"""
Position Sizing Calculator
===========================
Calculates exact position sizes based on risk parameters.
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from config import RISK_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position size calculation result."""
    position_size_usd: float       # Total position value in USD
    position_size_units: float     # Number of units (coins)
    risk_amount_usd: float         # Dollar amount at risk
    risk_pct: float                # Percentage of account at risk
    leverage_required: float       # Leverage needed (if any)
    margin_required: float         # Margin needed for position


class PositionSizer:
    """
    Calculates position sizes using the % Risk Model.

    Position Size = Risk Amount / (Entry - SL)

    Where:
    - Risk Amount = Account Balance * Risk Per Trade %
    - Entry - SL = Distance to stop loss
    """

    def __init__(
        self,
        account_balance: float,
        risk_per_trade_pct: float = None,
        max_portfolio_heat_pct: float = None
    ):
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct or RISK_CONFIG.risk_per_trade_pct
        self.max_portfolio_heat = max_portfolio_heat_pct or RISK_CONFIG.max_portfolio_heat

        # Current exposure tracking
        self._current_exposure: float = 0

    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.account_balance = new_balance

    def add_exposure(self, position_size_usd: float):
        """Add to current exposure."""
        self._current_exposure += position_size_usd

    def remove_exposure(self, position_size_usd: float):
        """Remove from current exposure."""
        self._current_exposure = max(0, self._current_exposure - position_size_usd)

    def get_current_exposure_pct(self) -> float:
        """Get current exposure as percentage of account."""
        return (self._current_exposure / self.account_balance) * 100

    def calculate_base_risk(self) -> float:
        """Calculate base risk amount in USD."""
        return self.account_balance * (self.risk_per_trade_pct / 100)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence_score: int = 100,
        win_streak_modifier: float = 1.0
    ) -> PositionSize:
        """
        Calculate position size based on risk parameters.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence_score: Signal confidence (0-100) - affects size
            win_streak_modifier: Modifier from circuit breaker (0.5-1.0)

        Returns:
            PositionSize with all calculations
        """
        # Base risk amount
        base_risk = self.calculate_base_risk()

        # Adjust for confidence score
        if confidence_score >= 85:
            confidence_modifier = 1.0      # Full size for high confidence
        elif confidence_score >= 75:
            confidence_modifier = 0.75     # 75% for medium-high
        elif confidence_score >= 65:
            confidence_modifier = 0.5      # 50% for medium
        else:
            confidence_modifier = 0.25     # 25% for low

        # Apply modifiers
        adjusted_risk = base_risk * confidence_modifier * win_streak_modifier

        # Calculate SL distance
        sl_distance = abs(entry_price - stop_loss)
        sl_distance_pct = (sl_distance / entry_price) * 100

        if sl_distance == 0:
            logger.warning("SL distance is 0, using 1% fallback")
            sl_distance = entry_price * 0.01
            sl_distance_pct = 1.0

        # Position size in units
        position_size_units = adjusted_risk / sl_distance

        # Position size in USD
        position_size_usd = position_size_units * entry_price

        # Check portfolio heat limit
        available_heat = self.max_portfolio_heat - self.get_current_exposure_pct()
        max_position_pct = min(available_heat, 100)  # Cap at 100%

        if max_position_pct <= 0:
            # No room for new positions
            return PositionSize(
                position_size_usd=0,
                position_size_units=0,
                risk_amount_usd=0,
                risk_pct=0,
                leverage_required=0,
                margin_required=0
            )

        max_position_usd = self.account_balance * (max_position_pct / 100)

        if position_size_usd > max_position_usd:
            # Scale down to fit heat limit
            scale_factor = max_position_usd / position_size_usd
            position_size_usd = max_position_usd
            position_size_units = position_size_usd / entry_price
            adjusted_risk = adjusted_risk * scale_factor

        # Calculate leverage (if position > balance)
        leverage_required = position_size_usd / self.account_balance
        if leverage_required < 1:
            leverage_required = 1  # No leverage needed

        # Margin required (assuming cross margin)
        margin_required = position_size_usd / leverage_required

        return PositionSize(
            position_size_usd=round(position_size_usd, 2),
            position_size_units=round(position_size_units, 8),
            risk_amount_usd=round(adjusted_risk, 2),
            risk_pct=round((adjusted_risk / self.account_balance) * 100, 2),
            leverage_required=round(leverage_required, 1),
            margin_required=round(margin_required, 2)
        )

    def validate_position(
        self,
        position: PositionSize,
        min_position_usd: float = 10
    ) -> Tuple[bool, str]:
        """
        Validate that position meets requirements.

        Returns:
            (is_valid, reason)
        """
        if position.position_size_usd < min_position_usd:
            return False, f"Position size ${position.position_size_usd:.2f} below minimum ${min_position_usd}"

        if position.risk_pct > self.risk_per_trade_pct * 1.5:
            return False, f"Risk {position.risk_pct:.1f}% exceeds max {self.risk_per_trade_pct * 1.5:.1f}%"

        new_exposure = self.get_current_exposure_pct() + (position.position_size_usd / self.account_balance * 100)
        if new_exposure > self.max_portfolio_heat:
            return False, f"Would exceed portfolio heat: {new_exposure:.1f}% > {self.max_portfolio_heat}%"

        return True, "OK"

    def calculate_partial_exit(
        self,
        original_size_units: float,
        exit_pct: float
    ) -> float:
        """Calculate units to exit for partial close."""
        return original_size_units * (exit_pct / 100)

    def get_sizing_summary(self) -> str:
        """Get current sizing status."""
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
📐 POSITION SIZING
━━━━━━━━━━━━━━━━━━━━━━━━━
Account: ${self.account_balance:,.0f}
Risk/Trade: {self.risk_per_trade_pct}%
Max Heat: {self.max_portfolio_heat}%
Current Exposure: {self.get_current_exposure_pct():.1f}%
Available: {self.max_portfolio_heat - self.get_current_exposure_pct():.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Singleton
_sizer: Optional[PositionSizer] = None


def get_position_sizer(account_balance: float = None) -> PositionSizer:
    """Get position sizer singleton."""
    global _sizer
    if _sizer is None:
        from config import DEFAULT_USER_SETTINGS
        balance = account_balance or DEFAULT_USER_SETTINGS.account_balance
        _sizer = PositionSizer(balance)
    return _sizer


def calculate_position(
    entry_price: float,
    stop_loss: float,
    confidence: int = 100
) -> PositionSize:
    """Convenience function for position calculation."""
    sizer = get_position_sizer()
    return sizer.calculate_position_size(entry_price, stop_loss, confidence)
