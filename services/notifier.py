"""Notification service for Telegram and Discord."""
import asyncio
import logging
from typing import Optional, Dict, Any, List
import aiohttp

logger = logging.getLogger(__name__)


class Notifier:
    """
    Multi-channel notification service.

    Supports:
    - Telegram (via existing bot)
    - Discord webhooks
    """

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize the notifier.

        Args:
            discord_webhook_url: Discord webhook URL for sending alerts
            session: Optional aiohttp session (will create one if not provided)
        """
        self.discord_webhook_url = discord_webhook_url
        self._session = session
        self._owns_session = False

    async def __aenter__(self) -> "Notifier":
        """Create session on context entry if needed."""
        if not self._session:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close session on context exit if we own it."""
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def send_discord(
        self,
        title: str,
        description: str,
        color: int = 0x00FF00,  # Green
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
    ) -> bool:
        """
        Send a message to Discord via webhook.

        Args:
            title: Embed title
            description: Embed description
            color: Embed color (hex)
            fields: Optional list of embed fields
            footer: Optional footer text

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.discord_webhook_url:
            logger.debug("Discord webhook not configured, skipping")
            return False

        if not self._session:
            logger.error("Session not initialized")
            return False

        # Build embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }

        if fields:
            embed["fields"] = fields

        if footer:
            embed["footer"] = {"text": footer}

        payload = {
            "embeds": [embed]
        }

        try:
            async with self._session.post(
                self.discord_webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 204:
                    logger.info("Discord notification sent successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Discord webhook failed: {response.status} - {error_text}")
                    return False

        except asyncio.TimeoutError:
            logger.error("Discord webhook timeout")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Discord webhook error: {e}")
            return False

    async def send_signal_alert(
        self,
        symbol: str,
        signal_type: str,
        confidence: str,
        price: float,
        indicators: Dict[str, Any],
        analysis: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send a trading signal alert to all configured channels.

        Args:
            symbol: Trading pair symbol
            signal_type: "BULLISH" or "BEARISH"
            confidence: "HIGH", "MEDIUM", or "LOW"
            price: Current price
            indicators: Dict of indicator results
            analysis: Optional AI analysis text

        Returns:
            Dict mapping channel name to success status
        """
        results = {}

        # Prepare Discord message
        if self.discord_webhook_url:
            # Determine color
            if signal_type == "BULLISH":
                color = 0x00FF00  # Green
                emoji = "🟢"
            else:
                color = 0xFF0000  # Red
                emoji = "🔴"

            # Build fields from indicators
            fields = []

            # Add confluence info
            bullish_count = sum(1 for v in indicators.values() if v.get("signal") == "BULLISH")
            bearish_count = sum(1 for v in indicators.values() if v.get("signal") == "BEARISH")
            total_count = len(indicators)

            fields.append({
                "name": "Confluence",
                "value": f"**{bullish_count}** Bullish / **{bearish_count}** Bearish (of {total_count})",
                "inline": True
            })

            fields.append({
                "name": "Price",
                "value": f"${price:,.6f}" if price < 1 else f"${price:,.2f}",
                "inline": True
            })

            fields.append({
                "name": "Confidence",
                "value": confidence,
                "inline": True
            })

            # Add key indicators
            key_indicators = ["rsi", "macd", "ema", "market_structure"]
            for ind_name in key_indicators:
                if ind_name in indicators:
                    ind = indicators[ind_name]
                    signal = ind.get("signal", "?")
                    value = ind.get("value", "?")
                    signal_emoji = "+" if signal == "BULLISH" else ("-" if signal == "BEARISH" else "o")

                    fields.append({
                        "name": f"[{signal_emoji}] {ind_name.upper()}",
                        "value": f"{value} ({signal})",
                        "inline": True
                    })

            title = f"{emoji} {signal_type} Signal: {symbol}"
            description = analysis[:500] + "..." if analysis and len(analysis) > 500 else (analysis or "")

            results["discord"] = await self.send_discord(
                title=title,
                description=description,
                color=color,
                fields=fields,
                footer="Scalp Signal Bot | Multi-Indicator Scanner"
            )

        return results

    def format_telegram_message(
        self,
        symbol: str,
        signal_type: str,
        confidence: str,
        price: float,
        indicators: Dict[str, Any],
        analysis: Optional[str] = None,
    ) -> str:
        """
        Format a signal alert for Telegram.

        Args:
            symbol: Trading pair symbol
            signal_type: "BULLISH" or "BEARISH"
            confidence: "HIGH", "MEDIUM", or "LOW"
            price: Current price
            indicators: Dict of indicator results
            analysis: Optional AI analysis text

        Returns:
            Formatted Telegram message
        """
        if signal_type == "BULLISH":
            emoji = "+"
            direction = "LONG"
        else:
            emoji = "-"
            direction = "SHORT"

        # Count confluence
        bullish_count = sum(1 for v in indicators.values() if v.get("signal") == "BULLISH")
        bearish_count = sum(1 for v in indicators.values() if v.get("signal") == "BEARISH")

        lines = [
            f"[{emoji}] **{signal_type} SIGNAL** [{emoji}]",
            f"",
            f"Symbol: **{symbol}**",
            f"Price: ${price:,.6f}" if price < 1 else f"Price: ${price:,.2f}",
            f"Direction: {direction}",
            f"Confidence: {confidence}",
            f"",
            f"**Indicator Confluence:** {bullish_count}+ / {bearish_count}-",
            f"",
        ]

        # Add key indicator values
        key_indicators = ["rsi", "macd", "ema", "market_structure", "volume", "cvd"]
        for ind_name in key_indicators:
            if ind_name in indicators:
                ind = indicators[ind_name]
                signal = ind.get("signal", "?")
                value = ind.get("value", "?")
                strength = ind.get("strength", 0)
                signal_marker = "+" if signal == "BULLISH" else ("-" if signal == "BEARISH" else "o")

                if isinstance(value, float):
                    value = f"{value:.2f}"

                lines.append(f"[{signal_marker}] {ind_name.upper()}: {value} (Str: {strength})")

        if analysis:
            lines.append("")
            lines.append("**AI Analysis:**")
            # Truncate if too long
            if len(analysis) > 300:
                analysis = analysis[:300] + "..."
            lines.append(analysis)

        return "\n".join(lines)
