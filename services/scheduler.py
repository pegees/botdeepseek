"""Scheduled alert service using APScheduler."""
import logging
from typing import Optional, Set

from telegram.ext import Application
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    ALERT_HOURS,
    ALERT_MIN_CONFIDENCE,
    DEFAULT_TIMEFRAME,
)
from core.binance import BinanceClient
from core.deepseek import DeepSeekClient
from models.signal import Signal
from services.formatter import SignalFormatter
from services.history import history_service

logger = logging.getLogger(__name__)

# Confidence levels for filtering
CONFIDENCE_ORDER = ["LOW", "MEDIUM", "HIGH"]


class AlertScheduler:
    """Manages scheduled market scans and alert delivery."""

    def __init__(self, app: Application):
        """
        Initialize the alert scheduler.

        Args:
            app: Telegram Application instance for sending messages
        """
        self.app = app
        self.scheduler = AsyncIOScheduler()
        self._enabled_users: Set[int] = set()
        self._running = False

    @property
    def enabled_users(self) -> Set[int]:
        """Get the set of users with alerts enabled."""
        return self._enabled_users

    async def start(self) -> None:
        """Start the scheduler and load users with alerts enabled."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        # Load users with alerts enabled from database
        try:
            user_ids = await history_service.get_users_with_alerts()
            self._enabled_users = set(user_ids)
            logger.info(f"Loaded {len(self._enabled_users)} users with alerts enabled")
        except Exception as e:
            logger.error(f"Failed to load alert users: {e}")
            self._enabled_users = set()

        # Build cron trigger for alert hours
        hours_str = ",".join(str(h) for h in ALERT_HOURS)

        self.scheduler.add_job(
            self._scan_and_alert,
            CronTrigger(hour=hours_str, minute=0),
            id="scheduled_market_scan",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        logger.info(f"Alert scheduler started. Alerts at hours: {hours_str}")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Alert scheduler stopped")

    async def enable_for_user(self, user_id: int) -> None:
        """
        Enable alerts for a user.

        Args:
            user_id: Telegram user ID
        """
        self._enabled_users.add(user_id)
        await history_service.set_alerts_enabled(user_id, True)
        logger.info(f"Enabled alerts for user {user_id}")

    async def disable_for_user(self, user_id: int) -> None:
        """
        Disable alerts for a user.

        Args:
            user_id: Telegram user ID
        """
        self._enabled_users.discard(user_id)
        await history_service.set_alerts_enabled(user_id, False)
        logger.info(f"Disabled alerts for user {user_id}")

    def is_enabled(self, user_id: int) -> bool:
        """Check if alerts are enabled for a user."""
        return user_id in self._enabled_users

    def _meets_confidence_threshold(self, signal: Signal) -> bool:
        """Check if signal meets minimum confidence threshold."""
        try:
            signal_level = CONFIDENCE_ORDER.index(signal.confidence.upper())
            min_level = CONFIDENCE_ORDER.index(ALERT_MIN_CONFIDENCE.upper())
            return signal_level >= min_level
        except ValueError:
            return False

    async def _scan_and_alert(self) -> None:
        """Scan ALL markets and send alerts to enabled users."""
        if not self._enabled_users:
            logger.info("Scheduled scan skipped: no users with alerts enabled")
            return

        logger.info(f"Running scheduled market scan for {len(self._enabled_users)} users")

        try:
            # Fetch market data for ALL pairs
            async with BinanceClient() as binance:
                all_pairs = await binance.fetch_all_usdt_perpetuals()
                logger.info(f"Scheduled scan: fetching data for {len(all_pairs)} pairs")
                market_data = await binance.get_market_data(all_pairs, DEFAULT_TIMEFRAME)

            if not market_data:
                logger.warning("Scheduled scan: failed to fetch market data")
                return

            # Analyze with DeepSeek
            async with DeepSeekClient() as deepseek:
                analysis = await deepseek.analyze(market_data, DEFAULT_TIMEFRAME)

            # Parse signal
            signal = Signal.from_deepseek_response(analysis, DEFAULT_TIMEFRAME)

            if not signal:
                logger.info("Scheduled scan: no clear setup found")
                return

            # Check confidence threshold
            if not self._meets_confidence_threshold(signal):
                logger.info(
                    f"Scheduled scan: signal confidence {signal.confidence} "
                    f"below threshold {ALERT_MIN_CONFIDENCE}"
                )
                return

            # Format alert message
            formatted = SignalFormatter.format_alert(signal, scheduled=True)

            # Send to all enabled users
            success_count = 0
            for user_id in list(self._enabled_users):
                try:
                    await self.app.bot.send_message(
                        chat_id=user_id,
                        text=formatted,
                        parse_mode="Markdown",
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send alert to user {user_id}: {e}")
                    # Optionally disable alerts for users we can't reach
                    # await self.disable_for_user(user_id)

            logger.info(
                f"Scheduled alert sent to {success_count}/{len(self._enabled_users)} users: "
                f"{signal.pair} {signal.direction}"
            )

            # Store the signal in history (without user_id since it's a broadcast)
            await history_service.add_signal(signal)

        except Exception as e:
            logger.error(f"Scheduled scan failed: {e}", exc_info=True)

    async def trigger_scan_now(self) -> Optional[Signal]:
        """
        Manually trigger a market scan (for testing).

        Returns:
            Signal if found, None otherwise
        """
        logger.info("Manual scan triggered")

        try:
            async with BinanceClient() as binance:
                all_pairs = await binance.fetch_all_usdt_perpetuals()
                market_data = await binance.get_market_data(all_pairs, DEFAULT_TIMEFRAME)

            if not market_data:
                return None

            async with DeepSeekClient() as deepseek:
                analysis = await deepseek.analyze(market_data, DEFAULT_TIMEFRAME)

            return Signal.from_deepseek_response(analysis, DEFAULT_TIMEFRAME)

        except Exception as e:
            logger.error(f"Manual scan failed: {e}")
            return None


# Global scheduler instance (initialized in bot.py)
scheduler: Optional[AlertScheduler] = None


def init_scheduler(app: Application) -> AlertScheduler:
    """Initialize the global scheduler instance."""
    global scheduler
    scheduler = AlertScheduler(app)
    return scheduler


def get_scheduler() -> Optional[AlertScheduler]:
    """Get the global scheduler instance."""
    return scheduler
