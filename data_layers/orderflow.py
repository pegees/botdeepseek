"""
Order Flow Data Layer
======================
Order book depth, trade aggression, funding rate, and open interest analysis.
Weight: 25% of confluence score.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

from config import ORDERFLOW_PARAMS, BINANCE_API_KEY

logger = logging.getLogger(__name__)

BINANCE_FUTURES_API = "https://fapi.binance.com"


@dataclass
class OrderBookData:
    """Order book depth analysis."""
    bid_volume: float        # Total bid volume
    ask_volume: float        # Total ask volume
    imbalance: float         # Bid/(Bid+Ask) ratio
    bid_wall_price: float    # Largest bid level
    ask_wall_price: float    # Largest ask level
    spread_pct: float        # Spread as percentage


@dataclass
class TradeFlowData:
    """Trade aggression analysis."""
    buy_volume: float        # Taker buy volume
    sell_volume: float       # Taker sell volume
    aggression: float        # Buy/(Buy+Sell) ratio
    large_buys: int          # Count of large buy orders
    large_sells: int         # Count of large sell orders


@dataclass
class FundingData:
    """Funding rate analysis."""
    current_rate: float      # Current funding rate
    predicted_rate: float    # Next predicted rate
    is_extreme: bool         # Is rate at extreme level
    bias: str               # "long_crowded", "short_crowded", "neutral"


@dataclass
class OIData:
    """Open interest analysis."""
    current_oi: float        # Current OI in contracts
    oi_change_pct: float     # % change in last period
    is_significant: bool     # Is change significant


@dataclass
class OrderFlowScore:
    """Combined order flow score."""
    total_score: float       # 0-100
    direction: str           # "long", "short", "neutral"
    summary: str

    orderbook: Optional[OrderBookData] = None
    trade_flow: Optional[TradeFlowData] = None
    funding: Optional[FundingData] = None
    oi: Optional[OIData] = None


class OrderFlowAnalyzer:
    """
    Analyzes order flow data for trading signals.

    Components:
    1. Order book depth (bid/ask imbalance)
    2. Trade aggression (taker buy/sell ratio)
    3. Funding rate (crowded positioning)
    4. Open interest changes
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookData]:
        """
        Fetch and analyze order book depth.
        """
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/depth"

        try:
            async with session.get(url, params={"symbol": symbol, "limit": limit}) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                # Parse bids and asks
                bids = [(float(b[0]), float(b[1])) for b in data["bids"]]
                asks = [(float(a[0]), float(a[1])) for a in data["asks"]]

                bid_volume = sum(qty for _, qty in bids)
                ask_volume = sum(qty for _, qty in asks)

                total = bid_volume + ask_volume
                imbalance = bid_volume / total if total > 0 else 0.5

                # Find walls (largest orders)
                bid_wall = max(bids, key=lambda x: x[1]) if bids else (0, 0)
                ask_wall = max(asks, key=lambda x: x[1]) if asks else (0, 0)

                # Calculate spread
                best_bid = bids[0][0] if bids else 0
                best_ask = asks[0][0] if asks else 0
                mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
                spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else 0

                return OrderBookData(
                    bid_volume=bid_volume,
                    ask_volume=ask_volume,
                    imbalance=imbalance,
                    bid_wall_price=bid_wall[0],
                    ask_wall_price=ask_wall[0],
                    spread_pct=spread_pct
                )

        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None

    async def fetch_recent_trades(self, symbol: str, limit: int = 500) -> Optional[TradeFlowData]:
        """
        Fetch and analyze recent trades for aggression.
        """
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/trades"

        try:
            async with session.get(url, params={"symbol": symbol, "limit": limit}) as response:
                if response.status != 200:
                    return None

                trades = await response.json()

                buy_volume = 0
                sell_volume = 0
                large_buys = 0
                large_sells = 0

                # Calculate average trade size for "large" threshold
                quantities = [float(t["qty"]) for t in trades]
                avg_qty = sum(quantities) / len(quantities) if quantities else 0
                large_threshold = avg_qty * 3

                for trade in trades:
                    qty = float(trade["qty"])
                    is_buyer_maker = trade["isBuyerMaker"]

                    if is_buyer_maker:
                        # Buyer is maker = taker is selling
                        sell_volume += qty
                        if qty > large_threshold:
                            large_sells += 1
                    else:
                        # Buyer is taker = buying aggression
                        buy_volume += qty
                        if qty > large_threshold:
                            large_buys += 1

                total = buy_volume + sell_volume
                aggression = buy_volume / total if total > 0 else 0.5

                return TradeFlowData(
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    aggression=aggression,
                    large_buys=large_buys,
                    large_sells=large_sells
                )

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return None

    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingData]:
        """
        Fetch and analyze funding rate.
        """
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/premiumIndex"

        try:
            async with session.get(url, params={"symbol": symbol}) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                current_rate = float(data.get("lastFundingRate", 0))
                predicted_rate = float(data.get("estimatedSettlePrice", 0))

                # Determine if extreme
                extreme_threshold = ORDERFLOW_PARAMS.funding_extreme_threshold
                is_extreme = abs(current_rate) >= extreme_threshold

                # Determine bias
                if current_rate > 0.001:  # 0.1%
                    bias = "long_crowded"
                elif current_rate < -0.001:
                    bias = "short_crowded"
                else:
                    bias = "neutral"

                return FundingData(
                    current_rate=current_rate,
                    predicted_rate=predicted_rate,
                    is_extreme=is_extreme,
                    bias=bias
                )

        except Exception as e:
            logger.error(f"Error fetching funding for {symbol}: {e}")
            return None

    async def fetch_open_interest(self, symbol: str) -> Optional[OIData]:
        """
        Fetch and analyze open interest.
        """
        session = await self._get_session()
        url = f"{BINANCE_FUTURES_API}/fapi/v1/openInterest"

        try:
            async with session.get(url, params={"symbol": symbol}) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                current_oi = float(data.get("openInterest", 0))

                # For OI change, we'd need historical data
                # Simplified: just return current OI
                return OIData(
                    current_oi=current_oi,
                    oi_change_pct=0,  # Would need history
                    is_significant=False
                )

        except Exception as e:
            logger.error(f"Error fetching OI for {symbol}: {e}")
            return None

    async def analyze(self, symbol: str) -> OrderFlowScore:
        """
        Analyze all order flow components and return combined score.
        """
        # Fetch all data concurrently
        orderbook, trades, funding, oi = await asyncio.gather(
            self.fetch_orderbook(symbol),
            self.fetch_recent_trades(symbol),
            self.fetch_funding_rate(symbol),
            self.fetch_open_interest(symbol),
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(orderbook, Exception):
            orderbook = None
        if isinstance(trades, Exception):
            trades = None
        if isinstance(funding, Exception):
            funding = None
        if isinstance(oi, Exception):
            oi = None

        # Calculate component scores
        scores = []
        summaries = []

        # 1. Order book score (40% of order flow)
        if orderbook:
            threshold = ORDERFLOW_PARAMS.imbalance_threshold
            if orderbook.imbalance >= threshold:
                ob_score = 60 + (orderbook.imbalance - threshold) * 100
                summaries.append(f"{orderbook.imbalance*100:.0f}% bid imbalance")
            elif orderbook.imbalance <= (1 - threshold):
                ob_score = 40 - (threshold - orderbook.imbalance) * 100
                summaries.append(f"{(1-orderbook.imbalance)*100:.0f}% ask imbalance")
            else:
                ob_score = 50
            scores.append(("orderbook", min(100, max(0, ob_score)), 0.4))
        else:
            scores.append(("orderbook", 50, 0.4))

        # 2. Trade flow score (35% of order flow)
        if trades:
            threshold = ORDERFLOW_PARAMS.aggression_threshold
            if trades.aggression >= threshold:
                tf_score = 60 + (trades.aggression - threshold) * 100
                summaries.append(f"Buy aggression {trades.aggression*100:.0f}%")
            elif trades.aggression <= (1 - threshold):
                tf_score = 40 - (threshold - trades.aggression) * 100
                summaries.append(f"Sell aggression {(1-trades.aggression)*100:.0f}%")
            else:
                tf_score = 50
            scores.append(("trade_flow", min(100, max(0, tf_score)), 0.35))
        else:
            scores.append(("trade_flow", 50, 0.35))

        # 3. Funding score (15% of order flow)
        if funding:
            if funding.bias == "long_crowded":
                # Contrarian: crowded longs = bearish
                fund_score = 30
                summaries.append(f"Longs crowded ({funding.current_rate*100:.3f}%)")
            elif funding.bias == "short_crowded":
                # Contrarian: crowded shorts = bullish
                fund_score = 70
                summaries.append(f"Shorts crowded ({funding.current_rate*100:.3f}%)")
            else:
                fund_score = 50
            scores.append(("funding", fund_score, 0.15))
        else:
            scores.append(("funding", 50, 0.15))

        # 4. OI score (10% of order flow)
        if oi and oi.is_significant:
            if oi.oi_change_pct > 0:
                oi_score = 60
                summaries.append(f"OI +{oi.oi_change_pct:.1f}%")
            else:
                oi_score = 40
                summaries.append(f"OI {oi.oi_change_pct:.1f}%")
        else:
            oi_score = 50
        scores.append(("oi", oi_score, 0.10))

        # Calculate weighted total
        total_score = sum(score * weight for _, score, weight in scores)

        # Determine direction
        if total_score >= 60:
            direction = "long"
        elif total_score <= 40:
            direction = "short"
        else:
            direction = "neutral"

        # Build summary
        summary = " | ".join(summaries[:2]) if summaries else "No data"

        return OrderFlowScore(
            total_score=round(total_score, 1),
            direction=direction,
            summary=summary,
            orderbook=orderbook if not isinstance(orderbook, Exception) else None,
            trade_flow=trades if not isinstance(trades, Exception) else None,
            funding=funding if not isinstance(funding, Exception) else None,
            oi=oi if not isinstance(oi, Exception) else None
        )


# Singleton
_analyzer: Optional[OrderFlowAnalyzer] = None


def get_orderflow_analyzer() -> OrderFlowAnalyzer:
    """Get the order flow analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OrderFlowAnalyzer()
    return _analyzer


async def analyze_orderflow(symbol: str) -> OrderFlowScore:
    """Convenience function to analyze order flow."""
    analyzer = get_orderflow_analyzer()
    return await analyzer.analyze(symbol)
