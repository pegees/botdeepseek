"""Binance Futures API client with retry logic and timeouts."""
import asyncio
import hashlib
import hmac
import time
import logging
from typing import Optional, Dict, List, Any

import aiohttp

from config import (
    BINANCE_FUTURES_URL,
    BINANCE_TIMEOUT,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    MAX_RETRIES,
    RETRY_DELAY,
    FALLBACK_PAIRS,
    BATCH_SIZE,
    BATCH_DELAY,
)
from core.exceptions import BinanceAPIError, RateLimitError, DataFetchError

logger = logging.getLogger(__name__)


class BinanceClient:
    """Async Binance Futures API client with connection pooling and retry logic."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=BINANCE_TIMEOUT)
        self._api_key = BINANCE_API_KEY
        self._api_secret = BINANCE_API_SECRET

    async def __aenter__(self) -> "BinanceClient":
        """Create session on context entry."""
        headers = {}
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key
        self._session = aiohttp.ClientSession(timeout=self._timeout, headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close session on context exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Make a request with exponential backoff retry logic.

        Args:
            endpoint: API endpoint (e.g., 'klines', 'ticker/24hr')
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            RateLimitError: If rate limit is exceeded
            BinanceAPIError: If request fails after all retries
        """
        if not self._session:
            raise BinanceAPIError("Client session not initialized. Use 'async with' context manager.")

        url = f"{BINANCE_FUTURES_URL}/{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After", "60")
                        raise RateLimitError(
                            f"Binance rate limit exceeded. Retry after {retry_after}s"
                        )

                    if response.status >= 500:
                        error_text = await response.text()
                        raise BinanceAPIError(f"Server error {response.status}: {error_text}")

                    # Client errors (4xx except 429)
                    error_text = await response.text()
                    raise BinanceAPIError(f"API error {response.status}: {error_text}")

            except asyncio.TimeoutError:
                last_error = BinanceAPIError(f"Request timed out after {BINANCE_TIMEOUT}s")
                logger.warning(f"Binance request timeout (attempt {attempt + 1}/{MAX_RETRIES}): {endpoint}")

            except aiohttp.ClientError as e:
                last_error = BinanceAPIError(f"Connection error: {e}")
                logger.warning(f"Binance connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

            except RateLimitError:
                raise  # Don't retry rate limits

            except BinanceAPIError as e:
                if "Server error" in str(e):
                    last_error = e
                    logger.warning(f"Binance server error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                else:
                    raise  # Don't retry client errors

            # Exponential backoff before retry
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

        raise BinanceAPIError(f"Failed after {MAX_RETRIES} attempts: {last_error}")

    async def fetch_klines(
        self, symbol: str, interval: str = "15m", limit: int = 50
    ) -> Optional[Dict]:
        """
        Fetch candlestick (kline) data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '15m', '1h')
            limit: Number of candles to fetch

        Returns:
            Dict with symbol and klines data, or None on failure
        """
        try:
            data = await self._request(
                "klines",
                {"symbol": symbol, "interval": interval, "limit": limit}
            )
            return {"symbol": symbol, "klines": data}
        except BinanceAPIError as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None

    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Fetch 24h ticker data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            Ticker data dict, or None on failure
        """
        try:
            return await self._request("ticker/24hr", {"symbol": symbol})
        except BinanceAPIError as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return None

    async def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current funding rate for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            Funding rate data dict, or None on failure
        """
        try:
            data = await self._request("fundingRate", {"symbol": symbol, "limit": 1})
            return data[0] if data else None
        except BinanceAPIError as e:
            logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    async def fetch_all_usdt_perpetuals(self) -> List[str]:
        """
        Fetch ALL USDT-margined perpetual trading pairs from Binance.

        Returns:
            List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT', ...])
            Typically returns 200+ pairs.
        """
        try:
            data = await self._request("exchangeInfo")

            pairs = [
                symbol["symbol"]
                for symbol in data.get("symbols", [])
                if (
                    symbol.get("contractType") == "PERPETUAL"
                    and symbol.get("status") == "TRADING"
                    and symbol.get("marginAsset") == "USDT"
                    and symbol.get("quoteAsset") == "USDT"
                )
            ]

            logger.info(f"Found {len(pairs)} USDT perpetual pairs on Binance")
            return sorted(pairs)

        except BinanceAPIError as e:
            logger.error(f"Failed to fetch trading pairs: {e}")
            logger.info(f"Using fallback list of {len(FALLBACK_PAIRS)} pairs")
            return FALLBACK_PAIRS

    async def fetch_all_tickers(self) -> Dict[str, Dict]:
        """
        Fetch 24h ticker data for ALL pairs in a single API call.
        Much faster than fetching individually.

        Returns:
            Dict mapping symbol to ticker data
        """
        try:
            data = await self._request("ticker/24hr")
            return {item["symbol"]: item for item in data}
        except BinanceAPIError as e:
            logger.error(f"Failed to fetch all tickers: {e}")
            return {}

    async def fetch_all_funding_rates(self) -> Dict[str, float]:
        """
        Fetch funding rates for ALL pairs in a single API call.

        Returns:
            Dict mapping symbol to funding rate (as percentage)
        """
        try:
            data = await self._request("premiumIndex")
            rates = {}
            for item in data:
                try:
                    rates[item["symbol"]] = float(item.get("lastFundingRate", 0)) * 100
                except (ValueError, TypeError):
                    rates[item["symbol"]] = 0.0
            return rates
        except BinanceAPIError as e:
            logger.error(f"Failed to fetch funding rates: {e}")
            return {}

    async def get_market_data(
        self, pairs: Optional[List[str]] = None, interval: str = "15m",
        progress_callback=None
    ) -> List[Dict]:
        """
        Fetch comprehensive market data for all pairs - OPTIMIZED VERSION.

        Uses bulk API endpoints to minimize requests:
        - 1 call for all tickers (instead of 200+)
        - 1 call for all funding rates (instead of 200+)
        - Klines only for top 50 candidates by volume

        Args:
            pairs: List of trading pairs (fetches all if None)
            interval: Timeframe for candle data
            progress_callback: Optional async callback(current, total) for progress updates

        Returns:
            List of market data dicts with calculated metrics

        Raises:
            DataFetchError: If no data could be fetched
        """
        # Fetch all pairs dynamically if not provided
        if pairs is None:
            pairs = await self.fetch_all_usdt_perpetuals()

        total_pairs = len(pairs)
        pairs_set = set(pairs)

        if progress_callback:
            await progress_callback(0, total_pairs)

        # STEP 1: Fetch ALL tickers and funding rates in bulk (2 API calls total)
        logger.info(f"Fetching bulk data for {total_pairs} pairs...")
        all_tickers, all_funding = await asyncio.gather(
            self.fetch_all_tickers(),
            self.fetch_all_funding_rates()
        )

        if not all_tickers:
            raise DataFetchError("Failed to fetch ticker data")

        if progress_callback:
            await progress_callback(total_pairs // 3, total_pairs)

        # STEP 2: Filter to our pairs and sort by volume to find top candidates
        candidates = []
        for symbol, ticker in all_tickers.items():
            if symbol not in pairs_set:
                continue
            try:
                volume = float(ticker.get("quoteVolume", 0))
                price_change = float(ticker.get("priceChangePercent", 0))
                candidates.append({
                    "symbol": symbol,
                    "ticker": ticker,
                    "funding": all_funding.get(symbol, 0.0),
                    "volume": volume,
                    "price_change": price_change,
                })
            except (ValueError, TypeError):
                continue

        # Sort by volume and take top 50 for detailed analysis
        candidates.sort(key=lambda x: x["volume"], reverse=True)
        top_candidates = candidates[:50]

        logger.info(f"Selected top {len(top_candidates)} pairs by volume for kline analysis")

        if progress_callback:
            await progress_callback(total_pairs // 2, total_pairs)

        # STEP 3: Fetch klines only for top candidates (50 API calls max)
        kline_tasks = [self.fetch_klines(c["symbol"], interval) for c in top_candidates]
        kline_results = await asyncio.gather(*kline_tasks)

        if progress_callback:
            await progress_callback(total_pairs * 3 // 4, total_pairs)

        # STEP 4: Process results
        market_data = []
        for i, candidate in enumerate(top_candidates):
            kline_data = kline_results[i]
            if not kline_data:
                continue

            try:
                processed = self._process_pair_data_fast(
                    candidate["symbol"],
                    kline_data,
                    candidate["ticker"],
                    candidate["funding"]
                )
                if processed:
                    market_data.append(processed)
            except Exception as e:
                logger.error(f"Error processing {candidate['symbol']}: {e}")
                continue

        if progress_callback:
            await progress_callback(total_pairs, total_pairs)

        if not market_data:
            raise DataFetchError("Failed to fetch market data for any pair")

        logger.info(f"Processed {len(market_data)} pairs for AI analysis")
        return market_data

    def _process_pair_data_fast(
        self,
        pair: str,
        kline_data: Dict,
        ticker_data: Dict,
        funding_rate: float,
    ) -> Optional[Dict]:
        """Process data using pre-fetched ticker and funding rate."""
        klines = kline_data.get("klines", [])
        if len(klines) < 10:
            return None

        # Parse recent candles
        recent_candles = []
        for k in klines[-10:]:
            if len(k) < 6:
                continue
            recent_candles.append({
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        if len(recent_candles) < 5:
            return None

        # Calculate metrics
        closes = [c["close"] for c in recent_candles]
        highs = [c["high"] for c in recent_candles]
        lows = [c["low"] for c in recent_candles]
        volumes = [c["volume"] for c in recent_candles]

        current_price = closes[-1]
        recent_high = max(highs)
        recent_low = min(lows)
        avg_volume = sum(volumes) / len(volumes)
        last_volume = volumes[-1]
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1.0

        # Price change from candles
        first_close = closes[0]
        if first_close > 0:
            price_change_pct = ((current_price - first_close) / first_close) * 100
        else:
            price_change_pct = 0.0

        return {
            "symbol": pair,
            "price": current_price,
            "price_change_pct": round(price_change_pct, 2),
            "volume_24h": float(ticker_data.get("quoteVolume", 0)),
            "volume_ratio": round(volume_ratio, 2),
            "recent_high": recent_high,
            "recent_low": recent_low,
            "funding_rate": round(funding_rate, 4),
            "recent_candles": recent_candles[-5:],
        }

    def _process_pair_data(
        self,
        pair: str,
        kline_data: Dict,
        ticker_data: Dict,
        funding_data: Optional[Dict],
    ) -> Optional[Dict]:
        """Process raw API data into structured market data."""
        klines = kline_data.get("klines", [])
        if len(klines) < 10:
            logger.warning(f"Insufficient kline data for {pair}: {len(klines)} candles")
            return None

        # Parse recent candles
        recent_candles = []
        for k in klines[-10:]:
            if len(k) < 6:
                continue
            recent_candles.append({
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        if len(recent_candles) < 5:
            return None

        # Calculate metrics
        closes = [c["close"] for c in recent_candles]
        highs = [c["high"] for c in recent_candles]
        lows = [c["low"] for c in recent_candles]
        volumes = [c["volume"] for c in recent_candles]

        current_price = closes[-1]
        recent_high = max(highs)
        recent_low = min(lows)
        avg_volume = sum(volumes) / len(volumes)
        last_volume = volumes[-1]
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1.0

        # Price change (avoid division by zero)
        first_close = closes[0]
        if first_close > 0:
            price_change_pct = ((current_price - first_close) / first_close) * 100
        else:
            price_change_pct = 0.0

        # Funding rate
        funding_rate = 0.0
        if funding_data and "fundingRate" in funding_data:
            try:
                funding_rate = float(funding_data["fundingRate"]) * 100
            except (ValueError, TypeError):
                pass

        return {
            "symbol": pair,
            "price": current_price,
            "price_change_pct": round(price_change_pct, 2),
            "volume_24h": float(ticker_data.get("quoteVolume", 0)),
            "volume_ratio": round(volume_ratio, 2),
            "recent_high": recent_high,
            "recent_low": recent_low,
            "funding_rate": round(funding_rate, 4),
            "recent_candles": recent_candles[-5:],
        }
