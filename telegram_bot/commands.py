"""
Telegram Bot Commands
======================
All command handlers for the Telegram bot.
"""
from typing import Optional
from datetime import datetime, timedelta

from telegram import Update
from telegram.ext import ContextTypes

from config import SIGNAL_THRESHOLDS, RISK_CONFIG
from database import get_database
from risk.risk_manager import get_risk_manager
from risk.circuit_breakers import get_circuit_breakers
from risk.position_sizing import get_position_sizer
from core.engine import get_engine


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
    help_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━
📚 SCALP BOT COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *TRADING*
/scan - Run market scan now
/signals - View recent signals
/open - View open positions

📈 *PERFORMANCE*
/stats - Today's performance
/pnl - P&L summary
/winrate - Win rate stats
/drawdown - Drawdown info

⚙️ *SETTINGS*
/set - Configure settings
/config - View configuration
/heat - Portfolio heat

🔒 *CONTROLS*
/status - Bot status
/pause - Pause signals
/resume - Resume signals
/breakers - Circuit breaker status

━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


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


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pnl command - P&L summary."""
    db = get_database()
    stats = db.get_today_stats()
    rm = get_risk_manager()

    pnl_usd = rm.account_balance * (stats['total_pnl_pct'] / 100)
    emoji = "💚" if stats['total_pnl_pct'] > 0 else "💔" if stats['total_pnl_pct'] < 0 else "➖"

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
{emoji} P&L SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━
Today: {stats['total_pnl_pct']:+.2f}% (${pnl_usd:+.2f})
Trades: {stats['total_trades']}
Wins: {stats['wins']}
Losses: {stats['losses']}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_winrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /winrate command."""
    db = get_database()
    stats = db.get_today_stats()

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 WIN RATE
━━━━━━━━━━━━━━━━━━━━━━━━━
Today: {stats['win_rate']:.1f}%
Wins: {stats['wins']}
Losses: {stats['losses']}
Total: {stats['total_trades']}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_drawdown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /drawdown command."""
    breakers = get_circuit_breakers()

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
📉 DRAWDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━
Daily P&L: {breakers.session.daily_pnl_pct:+.2f}%
Limit: -{RISK_CONFIG.max_portfolio_heat}%
Status: {'🟢 OK' if breakers.session.daily_pnl_pct > -RISK_CONFIG.max_portfolio_heat else '🔴 BREACHED'}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_heat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /heat command - portfolio heat."""
    breakers = get_circuit_breakers()
    rm = get_risk_manager()
    sizer = get_position_sizer()

    positions = breakers.session.open_positions
    heat = len(positions) * rm.risk_per_trade_pct

    positions_list = "\n".join(f"  • {p}" for p in positions) if positions else "  None"

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 PORTFOLIO HEAT
━━━━━━━━━━━━━━━━━━━━━━━━━
Open Positions: {len(positions)}
Risk/Trade: {rm.risk_per_trade_pct}%
Total Heat: {heat:.1f}%
Max Heat: {RISK_CONFIG.max_portfolio_heat}%
Available: {RISK_CONFIG.max_portfolio_heat - heat:.1f}%

Positions:
{positions_list}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_open(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /open command - view open positions."""
    db = get_database()
    positions = db.get_active_positions()

    if not positions:
        await update.message.reply_text("No open positions.")
        return

    lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━", "📊 OPEN POSITIONS", "━━━━━━━━━━━━━━━━━━━━━━━━━"]

    for pos in positions:
        dir_emoji = "🟢" if pos['direction'] == 'long' else "🔴"
        lines.append(f"{dir_emoji} {pos['symbol']}")
        lines.append(f"   Entry: ${pos['entry_price']:.4f}")
        lines.append(f"   SL: ${pos['stop_loss']:.4f}")
        lines.append(f"   Score: {pos['confluence_score']}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

    await update.message.reply_text("\n".join(lines))


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals command - view recent signals."""
    db = get_database()
    signals = db.get_recent_signals(limit=10)

    if not signals:
        await update.message.reply_text("No recent signals.")
        return

    lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━", "📡 RECENT SIGNALS", "━━━━━━━━━━━━━━━━━━━━━━━━━"]

    for sig in signals[:5]:
        dir_emoji = "🟢" if sig['direction'] == 'long' else "🔴"
        status_emoji = "✅" if sig['status'] == 'active' else "⏳"
        lines.append(f"{dir_emoji} {sig['symbol']} | {sig['confluence_score']}/100 {status_emoji}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

    await update.message.reply_text("\n".join(lines))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command - bot status."""
    engine = get_engine()
    status = engine.get_status()

    breaker_lines = []
    for b in status['circuit_breakers'][:5]:
        icon = '🔴' if b['triggered'] else '🟢'
        breaker_lines.append(f"{icon} {b['type'].replace('_', ' ').title()}")

    breakers_text = "\n".join(breaker_lines)

    msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 BOT STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━
Running: {'✅' if status['running'] else '❌'}
Can Trade: {'✅' if status['can_trade'] else '❌'}
Last Scan: {status['last_scan'] or 'Never'}
Signals/Hour: {status['signals_this_hour']}

Circuit Breakers:
{breakers_text}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    await update.message.reply_text(msg)


async def cmd_breakers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /breakers command - circuit breaker status."""
    breakers = get_circuit_breakers()
    summary = breakers.get_status_summary()
    await update.message.reply_text(summary)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pause command - pause signals."""
    breakers = get_circuit_breakers()
    breakers.session.paused_until = datetime.now() + timedelta(hours=24)
    await update.message.reply_text("⏸️ Signals PAUSED for 24 hours.\nUse /resume to restart.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /resume command - resume signals."""
    breakers = get_circuit_breakers()
    breakers.session.paused_until = None
    await update.message.reply_text("▶️ Signals RESUMED.")


async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /set command - configure settings."""
    args = context.args
    rm = get_risk_manager()
    db = get_database()
    user_id = str(update.effective_user.id)

    if not args:
        msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━
Balance: ${rm.account_balance:.0f}
Risk/Trade: {rm.risk_per_trade_pct}%
Min Score: {SIGNAL_THRESHOLDS.minimum_score}

Usage:
/set balance 5000
/set risk 1.5
/set score 70
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        await update.message.reply_text(msg)
        return

    if len(args) < 2:
        await update.message.reply_text("❌ Usage: /set <setting> <value>")
        return

    setting = args[0].lower()
    value = args[1]

    try:
        if setting == "balance":
            new_balance = float(value)
            if new_balance < 100:
                await update.message.reply_text("❌ Minimum balance is $100")
                return
            rm.update_balance(new_balance)
            db.save_user_settings(user_id, {"account_balance": new_balance})
            await update.message.reply_text(f"✅ Balance set to ${new_balance:,.0f}")

        elif setting == "risk":
            new_risk = float(value)
            if new_risk > 10:
                await update.message.reply_text("❌ Maximum risk is 10%")
                return
            if new_risk < 0.5:
                await update.message.reply_text("❌ Minimum risk is 0.5%")
                return
            rm.update_risk_pct(new_risk)
            db.save_user_settings(user_id, {"risk_per_trade_pct": new_risk})
            await update.message.reply_text(f"✅ Risk per trade set to {new_risk}%")

        elif setting == "score":
            new_score = int(value)
            if new_score < 50 or new_score > 95:
                await update.message.reply_text("❌ Score must be between 50-95")
                return
            db.save_user_settings(user_id, {"alert_minimum_score": new_score})
            await update.message.reply_text(f"✅ Minimum score set to {new_score}")

        else:
            await update.message.reply_text(f"❌ Unknown setting: {setting}")

    except ValueError:
        await update.message.reply_text(f"❌ Invalid value: {value}")


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /config command - show configuration."""
    from config import get_config_summary
    await update.message.reply_text(get_config_summary())


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /scan command - run market scan."""
    await update.message.reply_text("📡 Scanning market...")

    try:
        engine = get_engine()
        signals = await engine.run_once()

        if signals:
            # Send top signals
            for sig in signals[:3]:
                await update.message.reply_text(sig.formatted_message)

            from telegram_bot.formatter import format_scan_summary
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
                f"📊 Scan complete - no signals above threshold ({SIGNAL_THRESHOLDS.minimum_score})."
            )

    except Exception as e:
        await update.message.reply_text(f"❌ Scan failed: {str(e)}")
