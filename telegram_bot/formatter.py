"""
Signal Formatter - Professional Trading Signal Format
======================================================
Formats signals with full data breakdown and copy-ready prices.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from risk.risk_manager import RiskCalculation, TradeDirection
from data_layers.technical import TechnicalScore


@dataclass
class SignalData:
    """Complete signal data for formatting."""
    symbol: str
    direction: TradeDirection
    confidence_score: int        # 0-100 overall score
    entry_price: float
    risk_calc: RiskCalculation

    # Layer scores (0-100)
    ta_score: float
    ta_summary: str
    orderflow_score: float
    orderflow_summary: str
    onchain_score: float
    onchain_summary: str
    sentiment_score: float
    sentiment_summary: str

    # Meta
    timestamp: datetime = None
    expiry_minutes: int = 15


def get_star_rating(score: int) -> str:
    """Convert score to star rating."""
    if score >= 90:
        return "⭐⭐⭐⭐⭐"
    elif score >= 80:
        return "⭐⭐⭐⭐"
    elif score >= 70:
        return "⭐⭐⭐"
    elif score >= 60:
        return "⭐⭐"
    else:
        return "⭐"


def get_confidence_emoji(score: int) -> str:
    """Get emoji based on confidence."""
    if score >= 85:
        return "🔥"
    elif score >= 75:
        return "💪"
    elif score >= 65:
        return "👍"
    else:
        return "👀"


def format_price(price: float) -> str:
    """Format price with appropriate decimals."""
    if price >= 1000:
        return f"${price:.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def format_signal(data: SignalData) -> str:
    """
    Format a trading signal for Telegram.

    Example output:
    🟢 LONG — SOL/USDT 🔥
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 Score: 87/100 ⭐⭐⭐⭐

    💰 Entry: $178.35
    ├─ SL: $176.80 (0.87%)
    ├─ TP1: $179.70 → 40% | SL→BE
    ├─ TP2: $181.20 → 40% | Trail
    └─ TP3: $183.50 → 20%

    ⚖️ R:R 1:2.5 | Size: 4.2%
    💵 Risk: $50 (2% of $2500)

    📡 DATA
    ├─ TA: 91 (RSI 28, EMA bounce)
    ├─ Flow: 78 (67% bid imbalance)
    ├─ Chain: 72 ($2.3M withdrawn)
    └─ Sent: 68 (Fear 24)

    🤖 Data-driven. Zero emotion.
    """
    # Direction emoji and text
    if data.direction == TradeDirection.LONG:
        dir_emoji = "🟢"
        dir_text = "LONG"
    else:
        dir_emoji = "🔴"
        dir_text = "SHORT"

    conf_emoji = get_confidence_emoji(data.confidence_score)
    stars = get_star_rating(data.confidence_score)

    # Format symbol (add slash if needed)
    symbol = data.symbol
    if "USDT" in symbol and "/" not in symbol:
        symbol = symbol.replace("USDT", "/USDT")

    # Risk calculation
    rc = data.risk_calc

    # Build TP lines
    tp_lines = []
    for i, tp in enumerate(rc.take_profits):
        prefix = "├─" if i < len(rc.take_profits) - 1 else "└─"
        action = ""
        if tp.level == 1:
            action = " | SL→BE"
        elif tp.level == 2:
            action = " | Trail"

        tp_lines.append(
            f"{prefix} TP{tp.level}: {format_price(tp.price)} → {tp.size_pct*100:.0f}%{action}"
        )

    tp_section = "\n".join(tp_lines)

    # Build message
    msg = f"""
{dir_emoji} {dir_text} — {symbol} {conf_emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Score: {data.confidence_score}/100 {stars}

💰 Entry: {format_price(data.entry_price)}
├─ SL: {format_price(rc.stop_loss)} ({rc.sl_distance_pct:.2f}%)
{tp_section}

⚖️ R:R 1:{rc.rr_ratio:.1f} | Size: {format_price(rc.position_size_usd)}
💵 Risk: {format_price(rc.risk_amount_usd)} ({rc.risk_pct:.0f}%)

📡 DATA
├─ TA: {data.ta_score:.0f} ({data.ta_summary})
├─ Flow: {data.orderflow_score:.0f} ({data.orderflow_summary})
├─ Chain: {data.onchain_score:.0f} ({data.onchain_summary})
└─ Sent: {data.sentiment_score:.0f} ({data.sentiment_summary})

🤖 Data-driven. Zero emotion.
"""
    return msg.strip()


def format_quick_signal(
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    tp3: float,
    score: int
) -> str:
    """
    Quick format for copy-paste trading.
    Minimal format with just the essentials.
    """
    dir_emoji = "🟢" if direction.lower() == "long" else "🔴"

    return f"""
{dir_emoji} {symbol} | Score: {score}

Entry: {format_price(entry)}
SL: {format_price(sl)}
TP1: {format_price(tp1)}
TP2: {format_price(tp2)}
TP3: {format_price(tp3)}
"""


def format_scan_summary(
    pairs_scanned: int,
    signals_found: int,
    top_signals: List[Dict]
) -> str:
    """Format market scan summary."""
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📡 MARKET SCAN COMPLETE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Pairs scanned: {pairs_scanned}",
        f"Signals found: {signals_found}",
        ""
    ]

    if top_signals:
        lines.append("🔥 Top Signals:")
        for sig in top_signals[:5]:
            dir_emoji = "🟢" if sig.get("direction") == "long" else "🔴"
            lines.append(
                f"{dir_emoji} {sig['symbol']}: {sig['score']}/100"
            )

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_position_update(
    symbol: str,
    update_type: str,  # "TP1_HIT", "TP2_HIT", "TP3_HIT", "SL_HIT"
    entry_price: float,
    exit_price: float,
    pnl_pct: float
) -> str:
    """Format position update notification."""
    if "TP" in update_type:
        emoji = "✅"
        level = update_type.replace("_HIT", "")
    else:
        emoji = "❌"
        level = "SL"

    pnl_emoji = "💚" if pnl_pct > 0 else "💔"

    return f"""
{emoji} {level} HIT — {symbol}

Entry: {format_price(entry_price)}
Exit: {format_price(exit_price)}
{pnl_emoji} P&L: {pnl_pct:+.2f}%
"""


def format_circuit_breaker_alert(
    breaker_type: str,
    reason: str,
    action: str
) -> str:
    """Format circuit breaker alert."""
    return f"""
🔒 CIRCUIT BREAKER TRIGGERED
━━━━━━━━━━━━━━━━━━━━━━━━━
Type: {breaker_type}
Reason: {reason}
Action: {action}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def format_daily_summary(
    trades: int,
    wins: int,
    losses: int,
    pnl_pct: float,
    best_trade: Optional[Dict] = None,
    worst_trade: Optional[Dict] = None
) -> str:
    """Format daily trading summary."""
    win_rate = (wins / trades * 100) if trades > 0 else 0
    pnl_emoji = "💚" if pnl_pct > 0 else "💔" if pnl_pct < 0 else "➖"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊 DAILY SUMMARY",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Trades: {trades} ({wins}W / {losses}L)",
        f"Win Rate: {win_rate:.1f}%",
        f"{pnl_emoji} P&L: {pnl_pct:+.2f}%",
    ]

    if best_trade:
        lines.append(f"🏆 Best: {best_trade['symbol']} +{best_trade['pnl']:.2f}%")

    if worst_trade:
        lines.append(f"📉 Worst: {worst_trade['symbol']} {worst_trade['pnl']:.2f}%")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_error(error_msg: str) -> str:
    """Format error message."""
    return f"""
⚠️ Error

{error_msg}

Please try again or contact support.
"""


def format_help() -> str:
    """Format help message."""
    return """
━━━━━━━━━━━━━━━━━━━━━━━━━
📚 SCALP BOT COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━

/scan - Run market scan
/stats - View today's stats
/pnl - View P&L summary
/heat - View portfolio heat
/pause - Pause/resume signals
/set - Configure settings
/help - Show this help

━━━━━━━━━━━━━━━━━━━━━━━━━
"""
