"""
WebSocket Streaming
====================
Real-time data streaming from Binance WebSocket.
"""
import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://fstream.binance.com/stream"


@dataclass
class KlineData:
    """Parsed kline/candlestick data."""
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    is_closed: bool


@dataclass
class TradeData:
    """Parsed trade data."""
    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool
    timestamp: datetime


class BinanceWebSocket:
    """
    Manages WebSocket connections to Binance.

    Streams:
    - Klines (candlesticks) for all qualified pairs
    - Trades for order flow analysis
    - Depth (order book) for market microstructure
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
        self._subscribed_pairs: Set[str] = set()
        self._reconnect_delay = 1
        self._max_reconnect_delay = 30

        # Callbacks
        self._on_kline: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_depth: Optional[Callable] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def connect(self, pairs: List[str], interval: str = "15m"):
        """Connect to Binance WebSocket and subscribe to streams."""
        self._running = True
        self._subscribed_pairs = set(pairs)

        while self._running:
            try:
                await self._connect_and_stream(pairs, interval)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

                if self._running:
                    logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )

    async def _connect_and_stream(self, pairs: List[str], interval: str):
        """Establish connection and process messages."""
        session = await self._get_session()

        # Build stream names
        streams = []
        for pair in pairs:
            symbol = pair.replace("/", "").lower()
            streams.append(f"{symbol}@kline_{interval}")

        stream_path = "/".join(streams[:50])  # Limit to 50 streams per connection
        url = f"{BINANCE_WS_URL}?streams={stream_path}"

        logger.info(f"Connecting to Binance WebSocket with {len(streams)} streams...")

        async with session.ws_connect(url) as ws:
            self._ws = ws
            self._reconnect_delay = 1  # Reset on successful connect
            logger.info("WebSocket connected")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed")
                    break

    async def _handle_message(self, data: str):
        """Parse and route WebSocket messages."""
        try:
            msg = json.loads(data)

            if "stream" not in msg:
                return

            stream = msg["stream"]
            payload = msg["data"]

            if "@kline_" in stream:
                kline = self._parse_kline(payload)
                if self._on_kline and kline.is_closed:
                    await self._on_kline(kline)

            elif "@trade" in stream:
                trade = self._parse_trade(payload)
                if self._on_trade:
                    await self._on_trade(trade)

            elif "@depth" in stream:
                if self._on_depth:
                    await self._on_depth(payload)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def _parse_kline(self, data: dict) -> KlineData:
        """Parse kline data from WebSocket message."""
        k = data["k"]
        return KlineData(
            symbol=data["s"],
            interval=k["i"],
            open_time=datetime.fromtimestamp(k["t"] / 1000),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=datetime.fromtimestamp(k["T"] / 1000),
            is_closed=k["x"]
        )

    def _parse_trade(self, data: dict) -> TradeData:
        """Parse trade data from WebSocket message."""
        return TradeData(
            symbol=data["s"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            is_buyer_maker=data["m"],
            timestamp=datetime.fromtimestamp(data["T"] / 1000)
        )

    def on_kline(self, callback: Callable):
        """Register callback for kline events."""
        self._on_kline = callback

    def on_trade(self, callback: Callable):
        """Register callback for trade events."""
        self._on_trade = callback

    def on_depth(self, callback: Callable):
        """Register callback for depth events."""
        self._on_depth = callback

    async def close(self):
        """Close the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton
_ws: Optional[BinanceWebSocket] = None


def get_websocket() -> BinanceWebSocket:
    """Get WebSocket singleton."""
    global _ws
    if _ws is None:
        _ws = BinanceWebSocket()
    return _ws
