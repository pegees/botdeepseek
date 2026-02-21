"""
Backtest Performance Metrics
==============================
Calculate comprehensive performance metrics from backtest results.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics."""
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0

    # P&L
    total_pnl_pct: float = 0
    avg_win_pct: float = 0
    avg_loss_pct: float = 0
    profit_factor: float = 0

    # Risk metrics
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    avg_rr_achieved: float = 0

    # Trade stats
    avg_hold_time_minutes: float = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Best/Worst
    best_trade_pnl: float = 0
    worst_trade_pnl: float = 0
    best_trade_pair: str = ""
    worst_trade_pair: str = ""

    # Outcome breakdown
    tp1_exits: int = 0
    tp2_exits: int = 0
    tp3_exits: int = 0
    sl_exits: int = 0
    expired_exits: int = 0
    max_hold_exits: int = 0

    # Monthly breakdown
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    # Minimum thresholds (from BUILD_INSTRUCTIONS)
    meets_requirements: bool = False
    requirement_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1f}%",
            "total_pnl": f"{self.total_pnl_pct:.2f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "max_drawdown": f"{self.max_drawdown_pct:.2f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "avg_rr": f"{self.avg_rr_achieved:.2f}",
            "avg_hold_time": f"{self.avg_hold_time_minutes:.0f} min",
            "max_consecutive_losses": self.max_consecutive_losses,
            "meets_requirements": self.meets_requirements,
            "failures": self.requirement_failures
        }

    def format_report(self) -> str:
        """Format as readable report."""
        status = "✅ PASSED" if self.meets_requirements else "❌ FAILED"

        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BACKTEST RESULTS {status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 PERFORMANCE
├─ Total Trades: {self.total_trades}
├─ Win Rate: {self.win_rate:.1f}%
├─ Profit Factor: {self.profit_factor:.2f}
├─ Total P&L: {self.total_pnl_pct:+.2f}%
├─ Avg Win: {self.avg_win_pct:+.2f}%
└─ Avg Loss: {self.avg_loss_pct:.2f}%

⚖️ RISK METRICS
├─ Max Drawdown: {self.max_drawdown_pct:.2f}%
├─ Sharpe Ratio: {self.sharpe_ratio:.2f}
├─ Sortino Ratio: {self.sortino_ratio:.2f}
└─ Avg R:R Achieved: {self.avg_rr_achieved:.2f}

📊 TRADE STATS
├─ Avg Hold Time: {self.avg_hold_time_minutes:.0f} min
├─ Max Consecutive Wins: {self.max_consecutive_wins}
├─ Max Consecutive Losses: {self.max_consecutive_losses}
├─ Best Trade: {self.best_trade_pair} {self.best_trade_pnl:+.2f}%
└─ Worst Trade: {self.worst_trade_pair} {self.worst_trade_pnl:.2f}%

🎯 EXIT BREAKDOWN
├─ TP1 Exits: {self.tp1_exits} ({self.tp1_exits/max(1,self.total_trades)*100:.0f}%)
├─ TP2 Exits: {self.tp2_exits} ({self.tp2_exits/max(1,self.total_trades)*100:.0f}%)
├─ TP3 Exits: {self.tp3_exits} ({self.tp3_exits/max(1,self.total_trades)*100:.0f}%)
├─ SL Exits: {self.sl_exits} ({self.sl_exits/max(1,self.total_trades)*100:.0f}%)
└─ Other: {self.expired_exits + self.max_hold_exits}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def calculate_metrics(trades: List, initial_capital: float) -> BacktestMetrics:
    """
    Calculate comprehensive metrics from trade list.

    Minimum requirements from BUILD_INSTRUCTIONS:
    - Win Rate: > 55%
    - Profit Factor: > 1.5
    - Max Drawdown: < 15%
    - Sharpe Ratio: > 1.2
    - Avg R:R Achieved: > 1.5
    - Max Consecutive Losses: < 8
    - Sample Size: > 200 trades
    """
    metrics = BacktestMetrics()

    if not trades:
        return metrics

    metrics.total_trades = len(trades)

    # Separate wins and losses
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.win_rate = len(wins) / len(trades) * 100 if trades else 0

    # P&L calculations
    all_pnl = [t.pnl_pct for t in trades]
    metrics.total_pnl_pct = sum(all_pnl)

    if wins:
        metrics.avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins)
        metrics.best_trade_pnl = max(t.pnl_pct for t in wins)
        best_trade = max(wins, key=lambda t: t.pnl_pct)
        metrics.best_trade_pair = best_trade.pair

    if losses:
        metrics.avg_loss_pct = sum(t.pnl_pct for t in losses) / len(losses)
        metrics.worst_trade_pnl = min(t.pnl_pct for t in losses)
        worst_trade = min(losses, key=lambda t: t.pnl_pct)
        metrics.worst_trade_pair = worst_trade.pair

    # Profit factor
    gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 1
    metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    equity_curve = [initial_capital]
    for t in trades:
        equity_curve.append(equity_curve[-1] * (1 + t.pnl_pct / 100))

    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    metrics.max_drawdown_pct = max_dd

    # Sharpe & Sortino ratios
    if len(all_pnl) > 1:
        pnl_array = np.array(all_pnl)
        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array)

        # Sharpe (assuming risk-free rate = 0)
        metrics.sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # Sortino (only downside deviation)
        downside = pnl_array[pnl_array < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1
        metrics.sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Average R:R achieved
    rr_values = [t.rr_achieved for t in trades if t.rr_achieved != 0]
    metrics.avg_rr_achieved = sum(rr_values) / len(rr_values) if rr_values else 0

    # Hold time
    hold_times = [t.hold_time_minutes for t in trades if t.hold_time_minutes > 0]
    metrics.avg_hold_time_minutes = sum(hold_times) / len(hold_times) if hold_times else 0

    # Consecutive wins/losses
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    for t in trades:
        if t.pnl_pct > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_loss_streak = max(max_loss_streak, abs(current_streak))

    metrics.max_consecutive_wins = max_win_streak
    metrics.max_consecutive_losses = max_loss_streak

    # Exit breakdown
    for t in trades:
        if t.outcome:
            if t.outcome.value == "tp1":
                metrics.tp1_exits += 1
            elif t.outcome.value == "tp2":
                metrics.tp2_exits += 1
            elif t.outcome.value == "tp3":
                metrics.tp3_exits += 1
            elif t.outcome.value == "sl":
                metrics.sl_exits += 1
            elif t.outcome.value == "expired":
                metrics.expired_exits += 1
            elif t.outcome.value == "max_hold":
                metrics.max_hold_exits += 1

    # Monthly breakdown
    for t in trades:
        if t.entry_time:
            month_key = t.entry_time.strftime("%Y-%m")
            if month_key not in metrics.monthly_returns:
                metrics.monthly_returns[month_key] = 0
            metrics.monthly_returns[month_key] += t.pnl_pct

    # Check requirements
    failures = []

    if metrics.win_rate <= 55:
        failures.append(f"Win rate {metrics.win_rate:.1f}% <= 55%")
    if metrics.profit_factor <= 1.5:
        failures.append(f"Profit factor {metrics.profit_factor:.2f} <= 1.5")
    if metrics.max_drawdown_pct >= 15:
        failures.append(f"Max drawdown {metrics.max_drawdown_pct:.2f}% >= 15%")
    if metrics.sharpe_ratio <= 1.2:
        failures.append(f"Sharpe ratio {metrics.sharpe_ratio:.2f} <= 1.2")
    if metrics.avg_rr_achieved <= 1.5:
        failures.append(f"Avg R:R {metrics.avg_rr_achieved:.2f} <= 1.5")
    if metrics.max_consecutive_losses >= 8:
        failures.append(f"Max consecutive losses {metrics.max_consecutive_losses} >= 8")
    if metrics.total_trades < 200:
        failures.append(f"Sample size {metrics.total_trades} < 200")

    metrics.requirement_failures = failures
    metrics.meets_requirements = len(failures) == 0

    return metrics
