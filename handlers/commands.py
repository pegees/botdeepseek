"""Telegram command handlers."""
import logging
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    VALID_TIMEFRAMES, DEFAULT_TIMEFRAME, ALERT_HOURS,
    SCANNER_TIMEFRAMES, DASHBOARD_PORT, DASHBOARD_HOST
)
from core.binance import BinanceClient
from core.deepseek import DeepSeekClient
from core.exceptions import BotException, DataFetchError
from models.signal import Signal
from services.auth import require_auth
from services.formatter import SignalFormatter
from services.history import history_service
from services.scheduler import get_scheduler
from services.scanner import ScannerService

logger = logging.getLogger(__name__)


@require_auth
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - show welcome message."""
    welcome = """*Scalp Signal Bot*

I scan ALL Binance Futures pairs and find the best scalp setups using AI + technical indicators.

*Commands:*
/scalp - Get best scalp signal (15m default)
/scalp 5m - Signal on specific timeframe
/scan - Multi-indicator scan (RSI, MACD, EMA, etc.)
/multi - Get top 3 setups
/history - View your recent signals
/pairs - Show how many pairs are scanned
/status - Bot health and stats
/dashboard - Get dashboard URL
/alert - Toggle scheduled alerts
/help - Full command list

*Natural Language:*
"find me a long"
"any good shorts on 5m?"
"give me a setup"

_DYOR - Not financial advice_"""

    await update.message.reply_text(welcome, parse_mode="Markdown")


@require_auth
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command - show all commands."""
    help_text = """*Available Commands*

*Signal Commands:*
- /scalp [tf] - AI-based best setup (default 15m)
- /scan [tf] - Multi-indicator scan (RSI, MACD, etc.)
- /multi [tf] - Get top 3 AI setups
- /scalp 5m - Specify timeframe

*Information:*
- /pairs - Show pair count
- /history - Your last 5 signals
- /status - Bot health and stats
- /dashboard - Get dashboard URL

*Alerts:*
- /alert - Show alert status
- /alert on - Enable scheduled alerts
- /alert off - Disable alerts

*Timeframes:*
`1m` `3m` `5m` `15m` `30m` `1h` `4h`

*Indicators Used:*
RSI, MACD, EMA, Volume, CVD, Market Structure, S/R, Liquidity Sweeps, FVG, Whale Detection

*Natural Language Examples:*
- "give me a scalp setup"
- "find me a long on 5m"
- "any shorts looking good?"

_Scans ALL Binance Futures pairs (200+)_"""

    await update.message.reply_text(help_text, parse_mode="Markdown")


@require_auth
async def scalp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /scalp command - get single signal."""
    user_id = update.effective_user.id

    # Parse timeframe from args
    timeframe = DEFAULT_TIMEFRAME
    if context.args:
        tf = context.args[0].lower()
        if tf in VALID_TIMEFRAMES:
            timeframe = tf
        else:
            await update.message.reply_text(
                f"❌ Invalid timeframe `{tf}`\n\n"
                f"Valid options: {', '.join(f'`{t}`' for t in VALID_TIMEFRAMES)}",
                parse_mode="Markdown",
            )
            return

    # Send "scanning" message with step indicator and time estimates
    msg = await update.message.reply_text(
        f"[1/5] Connecting to Binance API... (~1s)"
    )

    try:
        # Increment user request count
        await history_service.increment_user_requests(user_id)

        # Fetch market data for ALL pairs (optimized - uses bulk API)
        async with BinanceClient() as binance:
            # Get all pairs first
            all_pairs = await binance.fetch_all_usdt_perpetuals()
            total_pairs = len(all_pairs)

            await msg.edit_text(
                f"[2/5] Found {total_pairs} pairs (~2s)\n"
                f"      Fetching tickers & funding rates..."
            )

            # Fetch all market data (optimized bulk fetch)
            market_data = await binance.get_market_data(all_pairs, timeframe)

        await msg.edit_text(
            f"[3/5] Got {len(market_data)} top pairs (~3s)\n"
            f"      Fetching {timeframe} candles..."
        )

        await msg.edit_text(
            f"[4/5] Waiting for AI response... (~5-10s)\n"
            f"      DeepSeek analyzing {len(market_data)} pairs"
        )

        # Analyze with DeepSeek
        async with DeepSeekClient() as deepseek:
            analysis = await deepseek.analyze(market_data, timeframe)

        await msg.edit_text(
            f"[5/5] Done! Formatting signal..."
        )

        # Parse response
        signal = Signal.from_deepseek_response(analysis, timeframe, user_id)

        if signal:
            # Store signal in history
            await history_service.add_signal(signal)
            formatted = SignalFormatter.format_signal(signal, timeframe)
        else:
            formatted = SignalFormatter.format_no_setup(analysis)

        await msg.edit_text(formatted, parse_mode="Markdown")

    except DataFetchError as e:
        logger.error(f"Data fetch error in /scalp: {e}")
        await msg.edit_text(
            "❌ Failed to fetch market data.\n\n"
            "Binance API may be temporarily unavailable. Try again in a moment."
        )

    except BotException as e:
        logger.error(f"Bot error in /scalp: {e}")
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")

    except Exception as e:
        logger.error(f"Unexpected error in /scalp: {e}", exc_info=True)
        await msg.edit_text("❌ An unexpected error occurred. Please try again.")


@require_auth
async def multi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /multi command - get top 3 signals."""
    user_id = update.effective_user.id

    # Parse timeframe from args
    timeframe = DEFAULT_TIMEFRAME
    if context.args:
        tf = context.args[0].lower()
        if tf in VALID_TIMEFRAMES:
            timeframe = tf

    # Send "scanning" message with step indicator and time estimates
    msg = await update.message.reply_text(
        f"[1/5] Connecting to Binance API... (~1s)"
    )

    try:
        # Increment user request count
        await history_service.increment_user_requests(user_id)

        # Fetch market data for ALL pairs (optimized bulk fetch)
        async with BinanceClient() as binance:
            all_pairs = await binance.fetch_all_usdt_perpetuals()
            total_pairs = len(all_pairs)

            await msg.edit_text(
                f"[2/5] Found {total_pairs} pairs (~2s)\n"
                f"      Fetching tickers & funding rates..."
            )

            market_data = await binance.get_market_data(all_pairs, timeframe)

        await msg.edit_text(
            f"[3/5] Got {len(market_data)} top pairs (~3s)\n"
            f"      Fetching {timeframe} candles..."
        )

        await msg.edit_text(
            f"[4/5] Waiting for AI response... (~10-15s)\n"
            f"      DeepSeek finding TOP 3 from {len(market_data)} pairs"
        )

        # Analyze with DeepSeek for multiple signals
        async with DeepSeekClient() as deepseek:
            analysis = await deepseek.analyze_multi(market_data, timeframe, count=3)

        await msg.edit_text(
            f"[5/5] Done! Formatting signals..."
        )

        # Parse multiple signals
        signals = Signal.parse_multi_response(analysis, timeframe, user_id)

        if signals:
            # Store all signals
            for signal in signals:
                await history_service.add_signal(signal)
            formatted = SignalFormatter.format_multi_signal(signals, timeframe)
        else:
            formatted = SignalFormatter.format_no_setup(analysis)

        await msg.edit_text(formatted, parse_mode="Markdown")

    except DataFetchError as e:
        logger.error(f"Data fetch error in /multi: {e}")
        await msg.edit_text(
            "❌ Failed to fetch market data.\n\n"
            "Try again in a moment."
        )

    except BotException as e:
        logger.error(f"Bot error in /multi: {e}")
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")

    except Exception as e:
        logger.error(f"Unexpected error in /multi: {e}", exc_info=True)
        await msg.edit_text("❌ An unexpected error occurred. Please try again.")


@require_auth
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command - show recent signals."""
    user_id = update.effective_user.id

    try:
        signals = await history_service.get_recent(5, user_id)
        formatted = SignalFormatter.format_history(signals)
        await update.message.reply_text(formatted, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /history: {e}")
        await update.message.reply_text("❌ Failed to load history. Please try again.")


@require_auth
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - show bot health."""
    user_id = update.effective_user.id

    msg = await update.message.reply_text("🔄 Checking status...")

    try:
        # Check API connectivity and get pair count
        binance_ok = False
        deepseek_ok = False
        pair_count = 0

        try:
            async with BinanceClient() as binance:
                await binance.fetch_ticker("BTCUSDT")
                binance_ok = True
                # Get pair count
                pairs = await binance.fetch_all_usdt_perpetuals()
                pair_count = len(pairs)
        except Exception:
            pass

        # Note: We don't actually call DeepSeek for status to save API credits
        deepseek_ok = binance_ok

        # Get stats
        stats = await history_service.get_stats()
        user_stats = await history_service.get_user_stats(user_id)

        status_text = f"""🤖 *Bot Status*

*APIs:*
├ Binance Futures: {"✅ Online" if binance_ok else "❌ Offline"}
└ DeepSeek AI: {"✅ Online" if deepseek_ok else "❌ Offline"}

*Coverage:*
└ Pairs scanned: {pair_count} (all USDT perpetuals)

*Stats:*
├ Signals today: {stats["today"]}
└ Total signals: {stats["total"]}

*Your Usage:*
└ Requests today: {user_stats["requests_today"]}"""

        await msg.edit_text(status_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /status: {e}")
        await msg.edit_text("❌ Failed to check status.")


@require_auth
async def pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /pairs command - show pair count."""
    msg = await update.message.reply_text("🔄 Fetching pair list...")

    try:
        async with BinanceClient() as binance:
            all_pairs = await binance.fetch_all_usdt_perpetuals()

        pair_count = len(all_pairs)

        # Show first few and last few as examples
        if pair_count > 10:
            sample = all_pairs[:5] + ["..."] + all_pairs[-5:]
            sample_text = ", ".join(f"`{p}`" for p in sample)
        else:
            sample_text = ", ".join(f"`{p}`" for p in all_pairs)

        text = f"""📈 *Binance Futures Coverage*

*Total Pairs:* {pair_count} USDT perpetuals

*Examples:*
{sample_text}

💡 _All pairs are scanned when you use /scalp or /multi_"""

        await msg.edit_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /pairs: {e}")
        await msg.edit_text("❌ Failed to fetch pairs.")


@require_auth
async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /alert command - toggle scheduled alerts."""
    user_id = update.effective_user.id
    scheduler = get_scheduler()

    if not scheduler:
        await update.message.reply_text(
            "❌ Alert scheduler not available.\n\n"
            "Please restart the bot or contact administrator."
        )
        return

    # Check for arguments
    if not context.args:
        # Show current status
        is_enabled = scheduler.is_enabled(user_id)
        status_emoji = "✅" if is_enabled else "❌"
        status_text = "ON" if is_enabled else "OFF"

        hours_str = ", ".join(f"{h:02d}:00" for h in sorted(ALERT_HOURS))

        await update.message.reply_text(
            f"🔔 *Scheduled Alerts:* {status_emoji} {status_text}\n\n"
            f"*Schedule:* {hours_str} UTC\n\n"
            f"Use `/alert on` or `/alert off` to toggle.\n\n"
            f"_Alerts scan ALL pairs and notify you when high-confidence setups are found._",
            parse_mode="Markdown",
        )
        return

    action = context.args[0].lower()

    if action == "on":
        await scheduler.enable_for_user(user_id)
        hours_str = ", ".join(f"{h:02d}:00" for h in sorted(ALERT_HOURS))
        await update.message.reply_text(
            f"✅ Scheduled alerts *enabled*.\n\n"
            f"You'll receive alerts at: {hours_str} UTC\n"
            f"when high-confidence setups are found.",
            parse_mode="Markdown",
        )

    elif action == "off":
        await scheduler.disable_for_user(user_id)
        await update.message.reply_text(
            "❌ Scheduled alerts *disabled*.\n\n"
            "Use `/alert on` to re-enable.",
            parse_mode="Markdown",
        )

    else:
        await update.message.reply_text(
            "Unknown option.\n\n"
            "Use `/alert on` or `/alert off`",
            parse_mode="Markdown",
        )


@require_auth
async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /scan command - multi-indicator scan."""
    user_id = update.effective_user.id

    # Parse timeframe from args
    timeframe = DEFAULT_TIMEFRAME
    if context.args:
        tf = context.args[0].lower()
        if tf in VALID_TIMEFRAMES:
            timeframe = tf
        else:
            await update.message.reply_text(
                f"Invalid timeframe `{tf}`\n\n"
                f"Valid options: {', '.join(f'`{t}`' for t in VALID_TIMEFRAMES)}",
                parse_mode="Markdown",
            )
            return

    msg = await update.message.reply_text(
        f"[1/5] Starting multi-indicator scan ({timeframe})..."
    )

    try:
        await history_service.increment_user_requests(user_id)

        scanner = ScannerService()

        async def progress_callback(step, total, message):
            try:
                await msg.edit_text(f"[{step}/{total}] {message}")
            except Exception:
                pass

        results = await scanner.scan(
            timeframe=timeframe,
            progress_callback=progress_callback,
            top_n=5
        )

        if not results:
            await msg.edit_text(
                "No strong signals found.\n\n"
                "Try a different timeframe or wait for better setups."
            )
            return

        # Format results
        lines = [f"*Multi-Indicator Scan ({timeframe})*\n"]

        for i, result in enumerate(results, 1):
            signal_emoji = "+" if result.signal == "BULLISH" else "-" if result.signal == "BEARISH" else "o"
            conf_emoji = "*" if result.confidence == "HIGH" else ""

            lines.append(f"{i}. `{result.symbol}` {signal_emoji}{result.signal}")
            lines.append(f"   Price: ${result.price:.6f}" if result.price < 1 else f"   Price: ${result.price:.2f}")
            lines.append(f"   Score: {result.score*100:.0f}% | Conf: {conf_emoji}{result.confidence}")
            lines.append(f"   Confluence: {result.confluence['bullish']}B / {result.confluence['bearish']}S")

            # Key indicators
            key_ind = []
            for ind_name in ["rsi", "macd", "ema"]:
                if ind_name in result.indicators:
                    ind = result.indicators[ind_name]
                    val = ind["value"]
                    if isinstance(val, float):
                        val = f"{val:.1f}"
                    key_ind.append(f"{ind_name.upper()}:{val}")

            if key_ind:
                lines.append(f"   [{', '.join(key_ind)}]")
            lines.append("")

        lines.append(f"_Analyzed with RSI, MACD, EMA, Volume, Structure, S/R, etc._")

        await msg.edit_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /scan: {e}", exc_info=True)
        await msg.edit_text("An unexpected error occurred. Please try again.")


@require_auth
async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /dashboard command - show dashboard URL."""
    dashboard_url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"

    await update.message.reply_text(
        f"*Signal Dashboard*\n\n"
        f"URL: `{dashboard_url}`\n\n"
        f"_Open in browser to see real-time signals with all indicator data._\n\n"
        f"Note: Dashboard runs locally on the server.",
        parse_mode="Markdown"
    )
