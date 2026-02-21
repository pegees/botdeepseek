#!/usr/bin/env python3
"""
Scalp Signal Bot - Main Entry Point
=====================================
Professional-grade 15m crypto scalping system with 4-layer confluence scoring.

Usage:
    export TELEGRAM_TOKEN="your_token"
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"
    python main.py
"""
import asyncio
import logging
import sys
import signal
from datetime import datetime
from typing import List

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_TOKEN, validate_config, get_config_summary,
    SIGNAL_THRESHOLDS, LOG_LEVEL, LOG_FORMAT
)
from core.engine import get_engine, TradingEngine, SignalOutput
from core.scanner import get_scanner
from risk.circuit_breakers import get_circuit_breakers
from risk.risk_manager import get_risk_manager
from database import get_database
from telegram_bot.formatter import format_help, format_scan_summary

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


# ============================================================================
# Telegram Command Handlers
# ============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "🤖 *Scalp Signal Bot*\n\n"
        "Professional-grade 15m crypto scalping system.\n"
        "Data-driven. Zero emotion.\n\n"
        "Use /help to see available commands.",
        parse_mode="Markdown"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await update.message.reply_text(format_help())


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /scan command - run market scan."""
    await update.message.reply_text("📡 Scanning market...")

    try:
        engine = get_engine()
        signals = await engine.run_once()

        if signals:
            # Send top signals
            for sig in signals[:3]:  # Top 3
                await update.message.reply_text(sig.formatted_message)

            # Summary
            await update.message.reply_text(
                format_scan_summary(
                    pairs_scanned=len(engine.scanner._cache),
                    signals_found=len(signals),
                    top_signals=[
                        {"symbol": s.symbol, "direction": s.direction.value, "score": s.confluence.total_score}
                        for s in signals[:5]
                    ]
                )
            )
        else:
            await update.message.reply_text(
                "📊 Scan complete - no signals above threshold.\n"
                f"Current threshold: {SIGNAL_THRESHOLDS.minimum_score}"
            )

    except Exception as e:
        logger.error(f"Scan error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Scan failed: {str(e)}")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command - show today's stats."""
    db = get_database()
    stats = db.get_today_stats()

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TODAY'S STATS
━━━━━━━━━━━━━━━━━━━━━━━━━
Trades: {stats['total_trades']} ({stats['wins']}W / {stats['losses']}L)
Win Rate: {stats['win_rate']:.1f}%
Total P&L: {stats['total_pnl_pct']:+.2f}%
Avg P&L: {stats['avg_pnl_pct']:+.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_heat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /heat command - show portfolio heat."""
    breakers = get_circuit_breakers()
    rm = get_risk_manager()

    open_positions = breakers.session.open_positions
    daily_pnl = breakers.session.daily_pnl_pct

    heat = len(open_positions) * rm.risk_per_trade_pct

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 PORTFOLIO HEAT
━━━━━━━━━━━━━━━━━━━━━━━━━
Open Positions: {len(open_positions)}
Risk Per Trade: {rm.risk_per_trade_pct}%
Total Heat: {heat:.1f}%
Daily P&L: {daily_pnl:+.1f}%

Positions:
{chr(10).join(open_positions) if open_positions else "None"}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command - show bot status."""
    engine = get_engine()
    status = engine.get_status()

    breaker_summary = "\n".join([
        f"{'🔴' if b['triggered'] else '🟢'} {b['type']}"
        for b in status['circuit_breakers'][:5]
    ])

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 BOT STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━
Running: {'✅' if status['running'] else '❌'}
Can Trade: {'✅' if status['can_trade'] else '❌'}
Last Scan: {status['last_scan'] or 'Never'}
Signals/Hour: {status['signals_this_hour']}

Circuit Breakers:
{breaker_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pause command - toggle alerts."""
    # Toggle pause state
    breakers = get_circuit_breakers()

    if breakers.session.paused_until:
        breakers.session.paused_until = None
        await update.message.reply_text("✅ Alerts resumed")
    else:
        from datetime import timedelta
        breakers.session.paused_until = datetime.now() + timedelta(hours=24)
        await update.message.reply_text("⏸️ Alerts paused for 24h")


async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /set command - configure settings."""
    args = context.args

    if not args:
        rm = get_risk_manager()
        msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━
Balance: ${rm.account_balance:.0f}
Risk/Trade: {rm.risk_per_trade_pct}%

Usage:
/set balance 5000
/set risk 1.5
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        await update.message.reply_text(msg)
        return

    if len(args) < 2:
        await update.message.reply_text("❌ Usage: /set <setting> <value>")
        return

    setting = args[0].lower()
    value = args[1]

    rm = get_risk_manager()
    db = get_database()
    user_id = str(update.effective_user.id)

    try:
        if setting == "balance":
            new_balance = float(value)
            rm.update_balance(new_balance)
            db.save_user_settings(user_id, {"account_balance": new_balance})
            await update.message.reply_text(f"✅ Balance set to ${new_balance:.0f}")

        elif setting == "risk":
            new_risk = float(value)
            if new_risk > 10:
                await update.message.reply_text("❌ Risk cannot exceed 10%")
                return
            rm.update_risk_pct(new_risk)
            db.save_user_settings(user_id, {"risk_per_trade_pct": new_risk})
            await update.message.reply_text(f"✅ Risk per trade set to {new_risk}%")

        else:
            await update.message.reply_text(f"❌ Unknown setting: {setting}")

    except ValueError:
        await update.message.reply_text(f"❌ Invalid value: {value}")


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /config command - show configuration."""
    await update.message.reply_text(get_config_summary())


# ============================================================================
# Signal Callback
# ============================================================================

async def send_signals(signals: List[SignalOutput], application: Application):
    """Send signals to configured chat."""
    from config import TELEGRAM_CHAT_ID

    if not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID not configured, signals not sent")
        return

    for signal in signals[:3]:  # Max 3 signals per scan
        try:
            await application.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=signal.formatted_message
            )
        except Exception as e:
            logger.error(f"Failed to send signal: {e}")


# ============================================================================
# Main
# ============================================================================

async def post_init(application: Application):
    """Initialize after bot starts."""
    # Set bot commands
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help"),
        BotCommand("scan", "Run market scan"),
        BotCommand("stats", "Today's stats"),
        BotCommand("heat", "Portfolio heat"),
        BotCommand("status", "Bot status"),
        BotCommand("pause", "Pause/resume alerts"),
        BotCommand("set", "Configure settings"),
        BotCommand("config", "Show configuration"),
    ]
    await application.bot.set_my_commands(commands)

    # Start engine loop in background
    engine = get_engine()

    async def signal_callback(signals):
        await send_signals(signals, application)

    asyncio.create_task(engine.run_loop(signal_callback))

    logger.info("Bot initialized and engine started")


async def shutdown(application: Application):
    """Clean shutdown."""
    engine = get_engine()
    engine.stop()
    await engine.close()
    await get_scanner().close()
    get_database().close()
    logger.info("Shutdown complete")


def main():
    """Main entry point."""
    # Validate config
    if not validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Starting Scalp Signal Bot")
    logger.info("Professional-grade 15m crypto scalping system")
    logger.info("=" * 50)

    # Build application
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(shutdown)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("heat", cmd_heat))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("config", cmd_config))

    # Run
    logger.info("Bot is starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
