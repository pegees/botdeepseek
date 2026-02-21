"""Multi-indicator scanner service."""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

from indicators import IndicatorRegistry
from core.binance import BinanceClient

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Represents a scan result for a single symbol."""
    symbol: str
    price: float
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    score: float  # 0.0 to 1.0
    indicators: Dict[str, Dict[str, Any]]
    confluence: Dict[str, int]
    timeframe: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScannerService:
    """
    Multi-indicator scanner that orchestrates market analysis.

    Features:
    - Scans all USDT perpetual pairs
    - Calculates all registered indicators
    - Computes confluence scores
    - Ranks signals by strength
    - Supports multiple timeframes
    """

    # Weighting for score calculation
    SCORE_WEIGHTS = {
        "confluence": 0.30,  # How many indicators agree
        "strength": 0.25,    # Average indicator strength
        "volume": 0.20,      # Volume confirmation
        "key_signals": 0.25, # Key indicator signals (RSI, MACD, structure)
    }

    # Key indicators that carry more weight
    KEY_INDICATORS = ["rsi", "macd", "ema", "market_structure"]

    def __init__(
        self,
        min_volume_24h: float = 5_000_000,
        min_confluence: int = 3,
        min_score: float = 0.5,
    ):
        """
        Initialize the scanner.

        Args:
            min_volume_24h: Minimum 24h volume in USD to scan
            min_confluence: Minimum indicators agreeing for a signal
            min_score: Minimum score (0-1) to include in results
        """
        self.min_volume_24h = min_volume_24h
        self.min_confluence = min_confluence
        self.min_score = min_score

    async def scan(
        self,
        timeframe: str = "15m",
        pairs: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
        top_n: int = 10,
    ) -> List[ScanResult]:
        """
        Run a full market scan.

        Args:
            timeframe: Candle timeframe to analyze
            pairs: Optional list of pairs to scan (scans all if None)
            progress_callback: Optional async callback(step, total, message)
            top_n: Return only top N results

        Returns:
            List of ScanResult objects sorted by score
        """
        results = []

        async with BinanceClient() as client:
            # Step 1: Get pairs to scan
            if progress_callback:
                await progress_callback(1, 5, "Fetching trading pairs...")

            if pairs is None:
                pairs = await client.fetch_all_usdt_perpetuals()

            logger.info(f"Scanning {len(pairs)} pairs on {timeframe} timeframe")

            # Step 2: Fetch market data
            if progress_callback:
                await progress_callback(2, 5, f"Fetching market data for {len(pairs)} pairs...")

            market_data = await client.get_market_data(pairs=pairs, interval=timeframe)

            if not market_data:
                logger.error("No market data received")
                return []

            logger.info(f"Received data for {len(market_data)} pairs")

            # Step 3: Filter by volume
            if progress_callback:
                await progress_callback(3, 5, "Filtering by volume...")

            filtered_data = [
                d for d in market_data
                if d.get("volume_24h", 0) >= self.min_volume_24h
            ]

            logger.info(f"{len(filtered_data)} pairs pass volume filter (>${self.min_volume_24h:,.0f})")

            # Step 4: Calculate indicators and score
            if progress_callback:
                await progress_callback(4, 5, f"Analyzing {len(filtered_data)} pairs with indicators...")

            for data in filtered_data:
                try:
                    result = self._analyze_pair(data, timeframe)
                    if result and result.score >= self.min_score:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {data.get('symbol')}: {e}")
                    continue

            # Step 5: Sort and return top N
            if progress_callback:
                await progress_callback(5, 5, "Ranking signals...")

            results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Found {len(results)} signals above min score, returning top {top_n}")

            return results[:top_n]

    def _analyze_pair(self, data: Dict, timeframe: str) -> Optional[ScanResult]:
        """
        Analyze a single pair with all indicators.

        Args:
            data: Market data dict with candles
            timeframe: Timeframe being analyzed

        Returns:
            ScanResult or None if insufficient data
        """
        symbol = data.get("symbol")
        candles = data.get("recent_candles", [])
        price = data.get("price", 0)

        if len(candles) < 5:
            return None

        # Convert candles to expected format (add high/low if missing)
        formatted_candles = []
        for c in candles:
            formatted = {
                "open": c.get("open", c.get("close", 0)),
                "high": c.get("high", max(c.get("open", 0), c.get("close", 0))),
                "low": c.get("low", min(c.get("open", 0), c.get("close", 0))),
                "close": c.get("close", 0),
                "volume": c.get("volume", 0),
            }
            formatted_candles.append(formatted)

        # Calculate all indicators
        indicator_results = IndicatorRegistry.calculate_all(formatted_candles)

        if not indicator_results:
            return None

        # Convert to dict format for easier access
        indicators_dict = {}
        for name, result in indicator_results.items():
            indicators_dict[name] = {
                "value": result.value,
                "signal": result.signal,
                "strength": result.strength,
                "metadata": result.metadata,
            }

        # Calculate confluence
        confluence = IndicatorRegistry.get_confluence(indicator_results)

        # Determine overall signal
        signal = "NEUTRAL"
        if confluence["bullish"] >= self.min_confluence and confluence["bullish"] > confluence["bearish"]:
            signal = "BULLISH"
        elif confluence["bearish"] >= self.min_confluence and confluence["bearish"] > confluence["bullish"]:
            signal = "BEARISH"

        # Calculate score
        score = self._calculate_score(indicators_dict, confluence, data)

        # Determine confidence
        confidence = self._calculate_confidence(score, confluence)

        return ScanResult(
            symbol=symbol,
            price=price,
            signal=signal,
            confidence=confidence,
            score=score,
            indicators=indicators_dict,
            confluence=confluence,
            timeframe=timeframe,
            metadata={
                "volume_24h": data.get("volume_24h", 0),
                "volume_ratio": data.get("volume_ratio", 0),
                "funding_rate": data.get("funding_rate", 0),
            }
        )

    def _calculate_score(
        self,
        indicators: Dict[str, Dict],
        confluence: Dict[str, int],
        data: Dict
    ) -> float:
        """
        Calculate overall signal score.

        Score components:
        - Confluence: How many indicators agree
        - Strength: Average strength of agreeing indicators
        - Volume: Volume confirmation
        - Key signals: Important indicator alignment
        """
        total = confluence["bullish"] + confluence["bearish"] + confluence["neutral"]
        if total == 0:
            return 0.0

        # Confluence score (0-1)
        max_agreement = max(confluence["bullish"], confluence["bearish"])
        confluence_score = max_agreement / total if total > 0 else 0

        # Strength score (0-1) - average of agreeing indicators
        dominant_signal = "BULLISH" if confluence["bullish"] > confluence["bearish"] else "BEARISH"
        agreeing_strengths = [
            ind["strength"]
            for ind in indicators.values()
            if ind["signal"] == dominant_signal
        ]
        strength_score = sum(agreeing_strengths) / len(agreeing_strengths) if agreeing_strengths else 0

        # Volume score (0-1)
        volume_ratio = data.get("volume_ratio", 1.0)
        volume_score = min(1.0, volume_ratio / 2.0)  # 2x volume = max score

        # Key indicators score (0-1)
        key_agreeing = sum(
            1 for name in self.KEY_INDICATORS
            if name in indicators and indicators[name]["signal"] == dominant_signal
        )
        key_score = key_agreeing / len(self.KEY_INDICATORS)

        # Weighted final score
        score = (
            confluence_score * self.SCORE_WEIGHTS["confluence"] +
            strength_score * self.SCORE_WEIGHTS["strength"] +
            volume_score * self.SCORE_WEIGHTS["volume"] +
            key_score * self.SCORE_WEIGHTS["key_signals"]
        )

        return round(score, 3)

    def _calculate_confidence(self, score: float, confluence: Dict[str, int]) -> str:
        """Determine confidence level based on score and confluence."""
        max_agreement = max(confluence["bullish"], confluence["bearish"])

        if score >= 0.75 and max_agreement >= 6:
            return "HIGH"
        elif score >= 0.55 and max_agreement >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    async def scan_multi_timeframe(
        self,
        timeframes: List[str] = None,
        pairs: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
        top_n: int = 10,
    ) -> Dict[str, List[ScanResult]]:
        """
        Scan multiple timeframes and combine results.

        Args:
            timeframes: List of timeframes to scan
            pairs: Optional pairs to scan
            progress_callback: Progress callback
            top_n: Top N results per timeframe

        Returns:
            Dict mapping timeframe to results
        """
        if timeframes is None:
            timeframes = ["5m", "15m", "1h"]

        results = {}
        total_steps = len(timeframes) * 5

        for i, tf in enumerate(timeframes):
            async def tf_callback(step, total, msg):
                if progress_callback:
                    overall_step = i * 5 + step
                    await progress_callback(overall_step, total_steps, f"[{tf}] {msg}")

            results[tf] = await self.scan(
                timeframe=tf,
                pairs=pairs,
                progress_callback=tf_callback,
                top_n=top_n
            )

        return results

    def get_consensus_signals(
        self,
        multi_tf_results: Dict[str, List[ScanResult]],
        min_timeframes: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find symbols that have signals across multiple timeframes.

        Args:
            multi_tf_results: Results from scan_multi_timeframe
            min_timeframes: Minimum timeframes that must agree

        Returns:
            List of consensus signals with combined data
        """
        # Group by symbol
        symbol_signals: Dict[str, List[ScanResult]] = {}

        for tf, results in multi_tf_results.items():
            for result in results:
                if result.symbol not in symbol_signals:
                    symbol_signals[result.symbol] = []
                symbol_signals[result.symbol].append(result)

        # Find consensus
        consensus = []

        for symbol, signals in symbol_signals.items():
            if len(signals) < min_timeframes:
                continue

            # Check if signals agree on direction
            bullish_count = sum(1 for s in signals if s.signal == "BULLISH")
            bearish_count = sum(1 for s in signals if s.signal == "BEARISH")

            if bullish_count >= min_timeframes:
                direction = "BULLISH"
                agreeing = bullish_count
            elif bearish_count >= min_timeframes:
                direction = "BEARISH"
                agreeing = bearish_count
            else:
                continue

            # Combine scores
            avg_score = sum(s.score for s in signals) / len(signals)
            timeframes_agreeing = [s.timeframe for s in signals if s.signal == direction]

            consensus.append({
                "symbol": symbol,
                "signal": direction,
                "timeframes": timeframes_agreeing,
                "avg_score": round(avg_score, 3),
                "signals": signals,
            })

        # Sort by number of agreeing timeframes, then by score
        consensus.sort(key=lambda x: (len(x["timeframes"]), x["avg_score"]), reverse=True)

        return consensus
