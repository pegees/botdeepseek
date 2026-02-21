"""
Backtest Replay Engine
=======================
Replays historical data through the signal engine to validate profitability.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

from config import (
    RISK_CONFIG, SIGNAL_THRESHOLDS, SCANNER_CONFIG,
    PRIMARY_TIMEFRAME
)
from backtesting.metrics import BacktestMetrics, calculate_metrics

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"


class TradeOutcome(Enum):
    """Trade outcome types."""
    TP1 = "tp1"
    TP2 = "tp2"
    TP3 = "tp3"
    SL = "sl"
    EXPIRED = "expired"
    MAX_HOLD = "max_hold"


@dataclass
class BacktestTrade:
    """A single backtested trade."""
    pair: str
    direction: str           # "long" or "short"
    entry_price: float
    entry_time: datetime
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confluence_score: int

    # Outcome
    exit_price: float = 0
    exit_time: datetime = None
    outcome: TradeOutcome = None
    pnl_pct: float = 0
    rr_achieved: float = 0
    hold_time_minutes: int = 0

    # Fees and slippage
    slippage_pct: float = 0.05   # 0.05%
    fee_pct: float = 0.1          # 0.1% taker fee


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    pairs: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000
    risk_per_trade_pct: float = 2.0
    max_concurrent_positions: int = 5
    slippage_pct: float = 0.05
    fee_pct: float = 0.1


class BacktestEngine:
    """
    Replays historical data through the signal engine.

    Requirements from BUILD_INSTRUCTIONS:
    1. Must use SAME code as live engine (no separate backtest logic)
    2. Must account for slippage (0.05% per trade)
    3. Must account for fees (0.1% maker/taker)
    4. Must respect all circuit breakers
    5. Walk-forward optimization (train on 2 months, test on 1 month)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._kline_cache: Dict[str, List[Dict]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Fetch historical kline data from Binance."""
        session = await self._get_session()

        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        while current_start < end_ms:
            url = f"{BINANCE_API}/api/v3/klines"
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000
            }

            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch klines: {response.status}")
                        break

                    data = await response.json()
                    if not data:
                        break

                    for k in data:
                        all_klines.append({
                            "open_time": datetime.fromtimestamp(k[0] / 1000),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "close_time": datetime.fromtimestamp(k[6] / 1000)
                        })

                    # Move to next batch
                    current_start = int(data[-1][6]) + 1

                    # Rate limit
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching klines for {symbol}: {e}")
                break

        return all_klines

    async def load_data(self):
        """Load all historical data for configured pairs."""
        logger.info(f"Loading data for {len(self.config.pairs)} pairs...")

        for pair in self.config.pairs:
            klines = await self.fetch_historical_klines(
                pair,
                PRIMARY_TIMEFRAME,
                self.config.start_date,
                self.config.end_date
            )
            self._kline_cache[pair] = klines
            logger.info(f"Loaded {len(klines)} candles for {pair}")

    def simulate_trade(
        self,
        trade: BacktestTrade,
        future_candles: List[Dict]
    ) -> BacktestTrade:
        """
        Simulate trade outcome using future candles.

        Checks if SL or TPs are hit in sequence.
        """
        # Apply entry slippage
        if trade.direction == "long":
            actual_entry = trade.entry_price * (1 + trade.slippage_pct / 100)
        else:
            actual_entry = trade.entry_price * (1 - trade.slippage_pct / 100)

        trade.entry_price = actual_entry

        # Track TP fills
        tp1_filled = False
        remaining_position = 1.0

        max_candles = 8  # Max hold time

        for i, candle in enumerate(future_candles[:max_candles]):
            high = candle["high"]
            low = candle["low"]

            if trade.direction == "long":
                # Check SL first (assume SL hit on wick before TP)
                if low <= trade.stop_loss:
                    trade.outcome = TradeOutcome.SL
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = candle["close_time"]
                    break

                # Check TPs
                if not tp1_filled and high >= trade.tp1:
                    tp1_filled = True
                    remaining_position -= 0.4
                    # Move SL to breakeven
                    trade.stop_loss = trade.entry_price

                if tp1_filled and high >= trade.tp2:
                    if remaining_position > 0.3:
                        remaining_position -= 0.4
                        # Move SL to TP1
                        trade.stop_loss = trade.tp1

                if high >= trade.tp3:
                    trade.outcome = TradeOutcome.TP3
                    trade.exit_price = trade.tp3
                    trade.exit_time = candle["close_time"]
                    break

            else:  # short
                # Check SL first
                if high >= trade.stop_loss:
                    trade.outcome = TradeOutcome.SL
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = candle["close_time"]
                    break

                # Check TPs
                if not tp1_filled and low <= trade.tp1:
                    tp1_filled = True
                    remaining_position -= 0.4
                    trade.stop_loss = trade.entry_price

                if tp1_filled and low <= trade.tp2:
                    if remaining_position > 0.3:
                        remaining_position -= 0.4
                        trade.stop_loss = trade.tp1

                if low <= trade.tp3:
                    trade.outcome = TradeOutcome.TP3
                    trade.exit_price = trade.tp3
                    trade.exit_time = candle["close_time"]
                    break

        # If no outcome yet, close at current price (max hold)
        if trade.outcome is None:
            if i >= max_candles - 1:
                trade.outcome = TradeOutcome.MAX_HOLD
            else:
                trade.outcome = TradeOutcome.EXPIRED

            trade.exit_price = future_candles[min(i, len(future_candles)-1)]["close"]
            trade.exit_time = future_candles[min(i, len(future_candles)-1)]["close_time"]

        # Calculate P&L
        if trade.direction == "long":
            trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100

        # Subtract fees
        trade.pnl_pct -= trade.fee_pct * 2  # Entry + exit

        # Calculate R achieved
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance > 0:
            profit_distance = abs(trade.exit_price - trade.entry_price)
            trade.rr_achieved = profit_distance / sl_distance
            if trade.pnl_pct < 0:
                trade.rr_achieved = -trade.rr_achieved

        # Calculate hold time
        if trade.exit_time and trade.entry_time:
            trade.hold_time_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

        return trade

    async def run(self) -> BacktestMetrics:
        """
        Run the backtest.

        Returns metrics for analysis.
        """
        # Load data if not already loaded
        if not self._kline_cache:
            await self.load_data()

        trades: List[BacktestTrade] = []
        portfolio = self.config.initial_capital

        logger.info("Starting backtest simulation...")

        # Import here to avoid circular imports
        from data_layers.technical import get_technical_analyzer
        from risk.risk_manager import get_risk_manager

        ta = get_technical_analyzer()
        rm = get_risk_manager()

        for pair, klines in self._kline_cache.items():
            if len(klines) < 200:
                continue

            # Iterate through candles
            for i in range(200, len(klines) - 10):
                # Get data up to this point
                lookback = klines[i-200:i+1]

                opens = [k["open"] for k in lookback]
                highs = [k["high"] for k in lookback]
                lows = [k["low"] for k in lookback]
                closes = [k["close"] for k in lookback]
                volumes = [k["volume"] for k in lookback]

                # Run TA
                ta_result = ta.analyze(opens, highs, lows, closes, volumes)

                # Check if signal generated
                if ta_result.total_score < SIGNAL_THRESHOLDS.minimum_score:
                    continue

                if ta_result.direction.value == "neutral":
                    continue

                # Calculate risk
                entry_price = closes[-1]
                direction = "long" if ta_result.direction.value == "long" else "short"

                from risk.risk_manager import TradeDirection
                trade_dir = TradeDirection.LONG if direction == "long" else TradeDirection.SHORT

                risk_calc = rm.calculate_full_risk(
                    pair, entry_price, trade_dir, highs, lows, closes
                )

                # Create trade
                trade = BacktestTrade(
                    pair=pair,
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=klines[i]["close_time"],
                    stop_loss=risk_calc.stop_loss,
                    tp1=risk_calc.take_profits[0].price,
                    tp2=risk_calc.take_profits[1].price,
                    tp3=risk_calc.take_profits[2].price,
                    confluence_score=int(ta_result.total_score),
                    slippage_pct=self.config.slippage_pct,
                    fee_pct=self.config.fee_pct
                )

                # Simulate outcome
                future = klines[i+1:i+10]
                if len(future) < 5:
                    continue

                trade = self.simulate_trade(trade, future)
                trades.append(trade)

        logger.info(f"Backtest complete: {len(trades)} trades simulated")

        # Calculate metrics
        metrics = calculate_metrics(trades, self.config.initial_capital)

        return metrics


async def run_backtest(
    pairs: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000
) -> BacktestMetrics:
    """Convenience function to run a backtest."""
    config = BacktestConfig(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    engine = BacktestEngine(config)
    try:
        metrics = await engine.run()
        return metrics
    finally:
        await engine.close()
