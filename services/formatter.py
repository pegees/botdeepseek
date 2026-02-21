"""Response formatting for Telegram messages."""
from datetime import datetime
from typing import List, Optional

from models.signal import Signal


class SignalFormatter:
    """Format trading signals for Telegram display."""

    @staticmethod
    def format_signal(signal: Signal, timeframe: Optional[str] = None) -> str:
        """
        Format a single signal for Telegram with markdown.

        Args:
            signal: Signal object to format
            timeframe: Override timeframe display (uses signal.timeframe if None)

        Returns:
            Formatted string for Telegram
        """
        tf = timeframe or signal.timeframe

        # Direction emoji and color indicator
        if signal.is_long:
            direction_emoji = "🟢"
            direction_text = "LONG"
        else:
            direction_emoji = "🔴"
            direction_text = "SHORT"

        # Confidence indicator
        confidence_icons = {
            "HIGH": "🔥",
            "MEDIUM": "⚡",
            "LOW": "💨",
        }
        conf_icon = confidence_icons.get(signal.confidence, "⚡")

        # Format the message
        return f"""{direction_emoji} *{direction_text} {signal.pair}* ({tf})

📍 *Entry:* `{signal.entry:,.8g}`
🛑 *Stop Loss:* `{signal.stop_loss:,.8g}` ({signal.sl_percent}%)
🎯 *Take Profit:* `{signal.take_profit:,.8g}` ({signal.tp_percent}%)

📊 *Risk/Reward:* {signal.risk_reward}
{conf_icon} *Confidence:* {signal.confidence}

💡 *Reasoning:*
{signal.reasoning}

⏰ {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
⚠️ _DYOR - Not financial advice_"""

    @staticmethod
    def format_no_setup(reason: str = "") -> str:
        """
        Format a 'no setup found' response.

        Args:
            reason: Explanation of why no setup was found

        Returns:
            Formatted string for Telegram
        """
        base_message = "🔍 *No Clear Setup Found*"

        if reason:
            # Clean up the reason text
            reason = reason.replace("NO CLEAR SETUP", "").strip()
            reason = reason.lstrip(":").strip()
            if reason:
                base_message += f"\n\n{reason}"

        base_message += "\n\n💡 _Try again later or check a different timeframe._"

        return base_message

    @staticmethod
    def format_multi_signal(signals: List[Signal], timeframe: str) -> str:
        """
        Format multiple signals in a compact list format.

        Args:
            signals: List of Signal objects
            timeframe: Timeframe for display

        Returns:
            Formatted string for Telegram
        """
        if not signals:
            return SignalFormatter.format_no_setup()

        header = f"📊 *Top {len(signals)} Setups ({timeframe})*\n"
        header += "─" * 20 + "\n\n"

        entries = []
        for i, sig in enumerate(signals[:5], 1):  # Max 5 signals
            emoji = "🟢" if sig.is_long else "🔴"
            conf_badge = {
                "HIGH": "🔥",
                "MEDIUM": "⚡",
                "LOW": "💨",
            }.get(sig.confidence, "")

            entry = (
                f"*{i}. {emoji} {sig.pair}* - {sig.direction}\n"
                f"   Entry: `{sig.entry:,.8g}`\n"
                f"   SL: `{sig.stop_loss:,.8g}` | TP: `{sig.take_profit:,.8g}`\n"
                f"   R:R {sig.risk_reward} {conf_badge}"
            )
            entries.append(entry)

        footer = f"\n\n⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        footer += "\n⚠️ _DYOR - Not financial advice_"

        return header + "\n\n".join(entries) + footer

    @staticmethod
    def format_history(signals: List[Signal]) -> str:
        """
        Format signal history for display.

        Args:
            signals: List of historical Signal objects

        Returns:
            Formatted string for Telegram
        """
        if not signals:
            return "📜 *No signal history yet.*\n\nUse /scalp to get your first signal!"

        header = f"📜 *Recent Signals ({len(signals)})*\n"
        header += "─" * 20 + "\n\n"

        lines = []
        for sig in signals:
            emoji = "🟢" if sig.is_long else "🔴"
            time_str = sig.timestamp.strftime("%m/%d %H:%M")
            lines.append(
                f"{emoji} *{sig.pair}* {sig.direction} @ `{sig.entry:,.8g}`\n"
                f"   {time_str} | {sig.timeframe} | {sig.confidence}"
            )

        return header + "\n\n".join(lines)

    @staticmethod
    def format_status(
        binance_ok: bool,
        deepseek_ok: bool,
        signals_today: int,
        total_signals: int,
        user_requests: int = 0,
    ) -> str:
        """
        Format bot status message.

        Args:
            binance_ok: Binance API connectivity status
            deepseek_ok: DeepSeek API connectivity status
            signals_today: Number of signals generated today
            total_signals: Total signals generated
            user_requests: User's request count today

        Returns:
            Formatted string for Telegram
        """
        binance_status = "✅ Online" if binance_ok else "❌ Offline"
        deepseek_status = "✅ Online" if deepseek_ok else "❌ Offline"

        return f"""🤖 *Bot Status*

*APIs:*
├ Binance Futures: {binance_status}
└ DeepSeek AI: {deepseek_status}

*Stats:*
├ Signals today: {signals_today}
└ Total signals: {total_signals}

*Your Usage:*
└ Requests today: {user_requests}"""

    @staticmethod
    def format_pairs(pairs: List[str]) -> str:
        """
        Format the list of monitored pairs.

        Args:
            pairs: List of trading pair symbols

        Returns:
            Formatted string for Telegram
        """
        # Group pairs into rows of 4 for cleaner display
        rows = []
        for i in range(0, len(pairs), 4):
            row = pairs[i : i + 4]
            rows.append(" | ".join(f"`{p}`" for p in row))

        pairs_display = "\n".join(rows)

        return f"""📈 *Monitored Pairs ({len(pairs)})*

{pairs_display}

💡 _Use /scalp to analyze these pairs_"""

    @staticmethod
    def format_alert(signal: Signal, scheduled: bool = False) -> str:
        """
        Format an alert notification.

        Args:
            signal: Signal object to format
            scheduled: Whether this is a scheduled alert

        Returns:
            Formatted string for Telegram
        """
        prefix = "🔔 *Scheduled Alert*\n\n" if scheduled else "🔔 *Alert*\n\n"
        return prefix + SignalFormatter.format_signal(signal)
