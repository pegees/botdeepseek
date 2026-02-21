"""
Alert Dispatcher
=================
Handles sending various alert types to Telegram.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from telegram import Bot
from telegram.error import TelegramError

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from telegram_bot.formatter import (
    format_signal,
    format_circuit_breaker_alert,
    format_position_update,
    format_daily_summary,
    SignalData,
)

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """
    Dispatches alerts to Telegram.

    Alert types:
    - Trading signals
    - Circuit breaker alerts
    - Whale alerts
    - Position updates (TP/SL hit)
    - Daily performance reports
    """

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self._bot: Optional[Bot] = None
        self._message_queue: List[str] = []
        self._rate_limit_delay = 0.1  # 100ms between messages

    def _get_bot(self) -> Bot:
        """Get bot instance."""
        if self._bot is None:
            self._bot = Bot(token=self.token)
        return self._bot

    async def send_message(self, message: str, parse_mode: str = None) -> bool:
        """Send a message to the configured chat."""
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not configured")
            return False

        try:
            bot = self._get_bot()
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            await asyncio.sleep(self._rate_limit_delay)
            return True

        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_signal(self, signal_data: SignalData) -> bool:
        """Send a trading signal alert."""
        message = format_signal(signal_data)
        return await self.send_message(message)

    async def send_circuit_breaker_alert(
        self,
        breaker_type: str,
        reason: str,
        action: str = "Signals PAUSED"
    ) -> bool:
        """Send circuit breaker activation alert."""
        message = format_circuit_breaker_alert(breaker_type, reason, action)
        return await self.send_message(message)

    async def send_position_update(
        self,
        symbol: str,
        update_type: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float
    ) -> bool:
        """Send position update (TP/SL hit)."""
        message = format_position_update(
            symbol, update_type, entry_price, exit_price, pnl_pct
        )
        return await self.send_message(message)

    async def send_whale_alert(
        self,
        token: str,
        amount_usd: float,
        direction: str,
        from_label: str,
        to_label: str
    ) -> bool:
        """Send whale movement alert."""
        amount_str = f"${amount_usd/1e6:.1f}M" if amount_usd >= 1e6 else f"${amount_usd/1e3:.0f}K"

        message = f"""
🐋 WHALE ALERT — {token}
━━━━━━━━━━━━━━━━━━━
{amount_str} {direction}
From: {from_label}
To: {to_label}
"""
        return await self.send_message(message)

    async def send_daily_report(
        self,
        trades: int,
        wins: int,
        losses: int,
        pnl_pct: float,
        best_trade: Optional[Dict] = None,
        worst_trade: Optional[Dict] = None
    ) -> bool:
        """Send daily performance report."""
        message = format_daily_summary(
            trades, wins, losses, pnl_pct, best_trade, worst_trade
        )
        return await self.send_message(message)

    async def send_startup_message(self) -> bool:
        """Send bot startup notification."""
        message = f"""
🤖 SCALP BOT STARTED
━━━━━━━━━━━━━━━━━━━
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: Active
Mode: Live Signals

Data-driven. Zero emotion.
"""
        return await self.send_message(message)

    async def send_shutdown_message(self, reason: str = "Manual stop") -> bool:
        """Send bot shutdown notification."""
        message = f"""
🔴 SCALP BOT STOPPED
━━━━━━━━━━━━━━━━━━━
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reason: {reason}
"""
        return await self.send_message(message)

    async def send_error_alert(self, error: str, component: str = "Unknown") -> bool:
        """Send error alert to admin."""
        message = f"""
⚠️ ERROR ALERT
━━━━━━━━━━━━━━━━━━━
Component: {component}
Error: {error}
Time: {datetime.now().strftime('%H:%M:%S')}
"""
        return await self.send_message(message)

    async def send_macro_danger_alert(self, btc_change: float) -> bool:
        """Send macro danger alert (BTC crash)."""
        message = f"""
🚨 MACRO DANGER ALERT
━━━━━━━━━━━━━━━━━━━
BTC 1h Change: {btc_change:+.1f}%
Action: ALL SIGNALS PAUSED
Reason: Extreme market volatility

Use /resume to restart when stable.
"""
        return await self.send_message(message)


# Singleton
_dispatcher: Optional[AlertDispatcher] = None


def get_alert_dispatcher() -> AlertDispatcher:
    """Get alert dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = AlertDispatcher()
    return _dispatcher


# Convenience functions
async def send_signal_alert(signal_data: SignalData) -> bool:
    return await get_alert_dispatcher().send_signal(signal_data)


async def send_breaker_alert(breaker_type: str, reason: str) -> bool:
    return await get_alert_dispatcher().send_circuit_breaker_alert(breaker_type, reason)


async def send_whale_alert(token: str, amount_usd: float, direction: str) -> bool:
    return await get_alert_dispatcher().send_whale_alert(
        token, amount_usd, direction, "Unknown", "Unknown"
    )
