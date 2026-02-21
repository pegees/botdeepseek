"""
Main Trading Engine - 15m Candle Close Trigger
===============================================
Orchestrates the entire scanning and signal generation process.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

from config import (
    PRIMARY_TIMEFRAME, SCANNER_CONFIG, SIGNAL_THRESHOLDS,
    BINANCE_API_KEY, BINANCE_API_SECRET
)
from core.scanner import get_scanner, PairInfo
from core.confluence import get_confluence_engine, ConfluenceResult
from data_layers.technical import get_technical_analyzer, TechnicalScore, SignalDirection
from risk.risk_manager import get_risk_manager, RiskCalculation, TradeDirection
from risk.circuit_breakers import get_circuit_breakers
from telegram_bot.formatter import SignalData, format_signal

logger = logging.getLogger(__name__)

BINANCE_FUTURES_API = "https://fapi.binance.com"


@dataclass
class SignalOutput:
    """Complete signal ready for sending."""
    symbol: str
    direction: TradeDirection
    confluence: ConfluenceResult
    risk: RiskCalculation
    formatted_message: str
    timestamp: datetime


class TradingEngine:
    """
    Main trading engine that orchestrates:
    1. Market scanning
    2. Data collection (OHLCV)
    3. Technical analysis
    4. Confluence scoring
    5. Risk calculation
    6. Signal generation
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self.scanner = get_scanner()
        self.ta_analyzer = get_technical_analyzer()
        self.confluence_engine = get_confluence_engine()
        self.risk_manager = get_risk_manager()
        self.circuit_breakers = get_circuit_breakers()

        # State
        self._last_scan_time: Optional[datetime] = None
        self._running = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the engine and clean up."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        await self.scanner.close()

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200
    ) -> List[Dict]:
        """
        Fetch kline/candlestick data from Binance.

        Returns list of:
        {
            'open_time': int,
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float,
            'close_time': int
        }
        """
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch klines for {symbol}: {response.status}")
                    return []

                data = await response.json()

                return [
                    {
                        "open_time": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": k[6]
                    }
                    for k in data
                ]
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []

    async def analyze_pair(self, symbol: str) -> Optional[Tuple[TechnicalScore, List[Dict]]]:
        """
        Analyze a single pair with technical indicators.

        Returns:
            (TechnicalScore, klines) or None if failed
        """
        # Fetch OHLCV data
        klines = await self.fetch_klines(symbol, PRIMARY_TIMEFRAME)

        if len(klines) < 50:
            logger.debug(f"{symbol}: Insufficient kline data")
            return None

        # Extract OHLCV
        opens = [k["open"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        closes = [k["close"] for k in klines]
        volumes = [k["volume"] for k in klines]

        # Run technical analysis
        ta_score = self.ta_analyzer.analyze(opens, highs, lows, closes, volumes)

        return ta_score, klines

    async def generate_signal(
        self,
        symbol: str,
        pair_info: PairInfo
    ) -> Optional[SignalOutput]:
        """
        Generate a complete trading signal for a pair.
        """
        # Check circuit breakers first
        can_trade, breaker_statuses, threshold_adj = self.circuit_breakers.check_all(
            new_symbol=symbol
        )

        if not can_trade:
            triggered = [s for s in breaker_statuses if s.is_triggered]
            logger.debug(f"{symbol}: Blocked by circuit breaker - {triggered[0].reason}")
            return None

        # Analyze pair
        analysis = await self.analyze_pair(symbol)
        if analysis is None:
            return None

        ta_score, klines = analysis

        # Calculate confluence (with placeholder scores for other layers)
        # In production, these would come from orderflow, onchain, sentiment modules
        confluence = self.confluence_engine.calculate_confluence(
            symbol=symbol,
            timeframe=PRIMARY_TIMEFRAME,
            technical=ta_score,
            orderflow_score=50,    # Placeholder
            orderflow_summary="N/A",
            onchain_score=50,      # Placeholder
            onchain_summary="N/A",
            sentiment_score=50,    # Placeholder
            sentiment_summary="N/A",
            backtest_score=50,     # Placeholder
            backtest_summary="N/A"
        )

        # Check if should signal
        effective_threshold = SIGNAL_THRESHOLDS.minimum_score + threshold_adj
        should_send, reason = self.confluence_engine.should_signal(
            confluence, threshold_adj
        )

        if not should_send:
            logger.debug(f"{symbol}: No signal - {reason}")
            return None

        # Skip neutral signals
        if confluence.direction == SignalDirection.NEUTRAL:
            return None

        # Convert direction
        trade_direction = (
            TradeDirection.LONG
            if confluence.direction == SignalDirection.LONG
            else TradeDirection.SHORT
        )

        # Calculate risk
        closes = [k["close"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        entry_price = closes[-1]

        risk_calc = self.risk_manager.calculate_full_risk(
            symbol=symbol,
            entry_price=entry_price,
            direction=trade_direction,
            highs=highs,
            lows=lows,
            closes=closes
        )

        # Validate risk
        is_valid, validation_msg = self.risk_manager.validate_risk(risk_calc)
        if not is_valid:
            logger.debug(f"{symbol}: Invalid risk - {validation_msg}")
            return None

        # Create signal data
        signal_data = SignalData(
            symbol=symbol,
            direction=trade_direction,
            confidence_score=int(confluence.total_score),
            entry_price=entry_price,
            risk_calc=risk_calc,
            ta_score=confluence.ta_score,
            ta_summary=confluence.ta_summary,
            orderflow_score=confluence.orderflow_score,
            orderflow_summary=confluence.orderflow_summary,
            onchain_score=confluence.onchain_score,
            onchain_summary=confluence.onchain_summary,
            sentiment_score=confluence.sentiment_score,
            sentiment_summary=confluence.sentiment_summary,
            timestamp=datetime.now()
        )

        # Format message
        formatted = format_signal(signal_data)

        # Record signal in circuit breakers
        self.circuit_breakers.record_signal()
        self.circuit_breakers.add_position(symbol)

        return SignalOutput(
            symbol=symbol,
            direction=trade_direction,
            confluence=confluence,
            risk=risk_calc,
            formatted_message=formatted,
            timestamp=datetime.now()
        )

    async def scan_market(self) -> List[SignalOutput]:
        """
        Scan entire market and generate signals.

        Returns:
            List of SignalOutput for each qualified signal
        """
        start_time = datetime.now()
        signals: List[SignalOutput] = []

        # Get qualified pairs
        scan_result = await self.scanner.scan()
        logger.info(f"Scanning {len(scan_result.qualified_pairs)} pairs...")

        # Analyze pairs in batches
        batch_size = SCANNER_CONFIG.batch_size
        pairs = scan_result.qualified_pairs

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            # Analyze batch concurrently
            tasks = [
                self.generate_signal(pair.symbol, pair)
                for pair in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, SignalOutput):
                    signals.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error generating signal: {result}")

            # Small delay between batches to avoid rate limits
            await asyncio.sleep(0.1)

        # Sort by confidence score
        signals.sort(key=lambda s: s.confluence.total_score, reverse=True)

        scan_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Scan complete: {len(signals)} signals from "
            f"{len(pairs)} pairs in {scan_time:.1f}s"
        )

        self._last_scan_time = datetime.now()

        return signals

    async def run_once(self) -> List[SignalOutput]:
        """Run a single scan cycle."""
        return await self.scan_market()

    async def run_loop(self, callback=None):
        """
        Run continuous scanning loop.
        Triggers at each 15m candle close.

        Args:
            callback: Async function to call with signals
        """
        self._running = True
        logger.info("Starting trading engine loop...")

        while self._running:
            try:
                # Calculate time to next 15m candle
                now = datetime.now()
                minutes = now.minute
                seconds = now.second

                # Next 15m mark
                next_15m = 15 - (minutes % 15)
                if next_15m == 15:
                    next_15m = 0

                wait_seconds = (next_15m * 60) - seconds + SCANNER_CONFIG.scan_offset_seconds

                if wait_seconds > 0:
                    logger.info(f"Waiting {wait_seconds}s for next 15m candle...")
                    await asyncio.sleep(wait_seconds)

                # Run scan
                signals = await self.scan_market()

                # Call callback with signals
                if callback and signals:
                    await callback(signals)

            except asyncio.CancelledError:
                logger.info("Engine loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in engine loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

        logger.info("Trading engine loop stopped")

    def stop(self):
        """Stop the engine loop."""
        self._running = False

    def get_status(self) -> Dict:
        """Get engine status."""
        breakers = self.circuit_breakers
        can_trade, statuses, _ = breakers.check_all()

        return {
            "running": self._running,
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "can_trade": can_trade,
            "circuit_breakers": [
                {"type": s.breaker_type.value, "triggered": s.is_triggered, "reason": s.reason}
                for s in statuses
            ],
            "open_positions": breakers.session.open_positions,
            "daily_pnl": breakers.session.daily_pnl_pct,
            "signals_this_hour": breakers.session.signals_this_hour
        }


# Singleton
_engine: Optional[TradingEngine] = None


def get_engine() -> TradingEngine:
    """Get the trading engine singleton."""
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine


async def run_scan() -> List[SignalOutput]:
    """Convenience function to run a market scan."""
    engine = get_engine()
    return await engine.run_once()
