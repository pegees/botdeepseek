"""
Market Scanner - Pair Discovery & Pre-filtering
================================================
Filters 400+ Binance Futures pairs down to 30-50 qualified pairs.
"""
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

from config import SCANNER_CONFIG, BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

# Binance API endpoints
BINANCE_FUTURES_API = "https://fapi.binance.com"


@dataclass
class PairInfo:
    """Information about a trading pair."""
    symbol: str
    base_asset: str
    quote_asset: str
    price: float
    volume_24h: float              # USDT volume
    price_change_24h: float        # % change
    high_24h: float
    low_24h: float
    volatility: float              # (high-low)/price %
    trades_24h: int
    funding_rate: float
    open_interest: float
    last_updated: datetime

    @property
    def is_active(self) -> bool:
        """Check if pair has sufficient activity."""
        return (
            self.volume_24h >= SCANNER_CONFIG.min_volume_usdt_24h
            and self.trades_24h > 1000
        )

    @property
    def volatility_score(self) -> float:
        """Score based on volatility (higher = more opportunity)."""
        if self.volatility < 1:
            return 30
        elif self.volatility < 3:
            return 60
        elif self.volatility < 5:
            return 80
        elif self.volatility < 10:
            return 100
        else:
            return 50  # Too volatile, risky


@dataclass
class ScanResult:
    """Result from scanning pairs."""
    qualified_pairs: List[PairInfo]
    total_scanned: int
    scan_time_ms: float
    timestamp: datetime


class MarketScanner:
    """
    Scans Binance Futures market to find qualified trading pairs.
    Filters by volume, volatility, and excludes stablecoins.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, PairInfo] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_exchange_info(self) -> Dict:
        """Fetch exchange info (trading rules, symbols)."""
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/exchangeInfo"

        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch exchange info: {response.status}")
            return await response.json()

    async def _fetch_24h_tickers(self) -> List[Dict]:
        """Fetch 24h ticker data for all symbols."""
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"

        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch tickers: {response.status}")
            return await response.json()

    async def _fetch_funding_rates(self) -> Dict[str, float]:
        """Fetch current funding rates."""
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/premiumIndex"

        async with session.get(url) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch funding rates: {response.status}")
                return {}

            data = await response.json()
            return {
                item["symbol"]: float(item.get("lastFundingRate", 0))
                for item in data
            }

    async def _fetch_open_interest(self) -> Dict[str, float]:
        """Fetch open interest for all symbols."""
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/openInterest"

        # Note: This endpoint requires symbol parameter, so we fetch for top pairs only
        # In production, you'd batch this or use websocket
        return {}  # Simplified for now

    def _is_excluded(self, symbol: str) -> bool:
        """Check if symbol should be excluded."""
        # Exclude stablecoins
        for excluded in SCANNER_CONFIG.excluded_pairs:
            if excluded in symbol:
                return True

        # Exclude leveraged tokens
        if any(x in symbol for x in ["UP", "DOWN", "BULL", "BEAR"]):
            return True

        return False

    def _parse_ticker(self, ticker: Dict, funding_rates: Dict) -> Optional[PairInfo]:
        """Parse ticker data into PairInfo."""
        try:
            symbol = ticker["symbol"]

            # Skip non-USDT pairs
            if not symbol.endswith("USDT"):
                return None

            # Skip excluded pairs
            if self._is_excluded(symbol):
                return None

            price = float(ticker["lastPrice"])
            high = float(ticker["highPrice"])
            low = float(ticker["lowPrice"])
            volume = float(ticker["quoteVolume"])  # USDT volume

            # Calculate volatility
            volatility = ((high - low) / price * 100) if price > 0 else 0

            return PairInfo(
                symbol=symbol,
                base_asset=symbol.replace("USDT", ""),
                quote_asset="USDT",
                price=price,
                volume_24h=volume,
                price_change_24h=float(ticker["priceChangePercent"]),
                high_24h=high,
                low_24h=low,
                volatility=volatility,
                trades_24h=int(ticker["count"]),
                funding_rate=funding_rates.get(symbol, 0),
                open_interest=0,  # Fetched separately if needed
                last_updated=datetime.now()
            )
        except (KeyError, ValueError) as e:
            logger.debug(f"Failed to parse ticker {ticker.get('symbol', 'unknown')}: {e}")
            return None

    async def scan(self, force_refresh: bool = False) -> ScanResult:
        """
        Scan market and return qualified pairs.

        Returns top pairs sorted by:
        1. Volume (primary)
        2. Volatility score (secondary)
        """
        start_time = datetime.now()

        # Check cache
        if not force_refresh and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_ttl:
                logger.debug("Using cached scan results")
                return ScanResult(
                    qualified_pairs=list(self._cache.values()),
                    total_scanned=len(self._cache),
                    scan_time_ms=0,
                    timestamp=self._cache_time
                )

        try:
            # Fetch all data concurrently
            tickers, funding_rates = await asyncio.gather(
                self._fetch_24h_tickers(),
                self._fetch_funding_rates()
            )

            # Parse and filter pairs
            all_pairs: List[PairInfo] = []
            for ticker in tickers:
                pair = self._parse_ticker(ticker, funding_rates)
                if pair and pair.is_active:
                    all_pairs.append(pair)

            # Sort by volume (descending)
            all_pairs.sort(key=lambda p: p.volume_24h, reverse=True)

            # Take top N pairs
            qualified = all_pairs[:SCANNER_CONFIG.max_pairs_to_analyze]

            # Update cache
            self._cache = {p.symbol: p for p in qualified}
            self._cache_time = datetime.now()

            scan_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                f"Scanned {len(tickers)} pairs → {len(qualified)} qualified "
                f"({scan_time:.0f}ms)"
            )

            return ScanResult(
                qualified_pairs=qualified,
                total_scanned=len(tickers),
                scan_time_ms=scan_time,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            # Return cached results if available
            if self._cache:
                return ScanResult(
                    qualified_pairs=list(self._cache.values()),
                    total_scanned=len(self._cache),
                    scan_time_ms=0,
                    timestamp=self._cache_time or datetime.now()
                )
            raise

    async def get_pair(self, symbol: str) -> Optional[PairInfo]:
        """Get info for a specific pair."""
        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]

        # Refresh scan if needed
        await self.scan()
        return self._cache.get(symbol)

    async def get_top_movers(self, limit: int = 10) -> List[PairInfo]:
        """Get top moving pairs by price change."""
        result = await self.scan()
        sorted_pairs = sorted(
            result.qualified_pairs,
            key=lambda p: abs(p.price_change_24h),
            reverse=True
        )
        return sorted_pairs[:limit]

    async def get_high_volume(self, limit: int = 10) -> List[PairInfo]:
        """Get highest volume pairs."""
        result = await self.scan()
        return result.qualified_pairs[:limit]

    async def get_high_volatility(self, limit: int = 10) -> List[PairInfo]:
        """Get highest volatility pairs (good for scalping)."""
        result = await self.scan()
        sorted_pairs = sorted(
            result.qualified_pairs,
            key=lambda p: p.volatility_score,
            reverse=True
        )
        return sorted_pairs[:limit]

    def get_scan_summary(self) -> str:
        """Get a summary of the last scan."""
        if not self._cache:
            return "No scan data available"

        pairs = list(self._cache.values())
        total_volume = sum(p.volume_24h for p in pairs)
        avg_volatility = sum(p.volatility for p in pairs) / len(pairs)

        top_5 = sorted(pairs, key=lambda p: p.volume_24h, reverse=True)[:5]
        top_list = "\n".join([
            f"  {p.symbol}: ${p.volume_24h/1e6:.1f}M | {p.price_change_24h:+.1f}%"
            for p in top_5
        ])

        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📡 MARKET SCAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Qualified pairs: {len(pairs)}
Total volume: ${total_volume/1e9:.2f}B
Avg volatility: {avg_volatility:.2f}%

🔥 Top 5 by Volume:
{top_list}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Singleton instance
_scanner: Optional[MarketScanner] = None


def get_scanner() -> MarketScanner:
    """Get the scanner singleton."""
    global _scanner
    if _scanner is None:
        _scanner = MarketScanner()
    return _scanner


async def scan_market() -> ScanResult:
    """Convenience function to scan market."""
    scanner = get_scanner()
    return await scanner.scan()
