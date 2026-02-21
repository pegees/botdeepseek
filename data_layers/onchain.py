"""
On-Chain Data Layer
====================
Whale alerts, exchange flows, and stablecoin movements.
Weight: 20% of confluence score.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

from config import ONCHAIN_PARAMS, WHALE_ALERT_API_KEY

logger = logging.getLogger(__name__)


@dataclass
class WhaleTransaction:
    """A single whale transaction."""
    amount_usd: float
    from_exchange: bool      # True if moving FROM exchange
    to_exchange: bool        # True if moving TO exchange
    symbol: str
    timestamp: datetime


@dataclass
class ExchangeFlows:
    """Exchange inflow/outflow data."""
    inflow_usd: float        # Money moving TO exchanges
    outflow_usd: float       # Money moving FROM exchanges
    net_flow: float          # Positive = inflow, Negative = outflow
    flow_bias: str           # "bullish", "bearish", "neutral"


@dataclass
class StablecoinFlow:
    """Stablecoin movement data."""
    inflow_usd: float        # Stablecoins moving to exchanges
    is_bullish: bool         # Large inflow = buying pressure


@dataclass
class OnChainScore:
    """Combined on-chain score."""
    total_score: float       # 0-100
    direction: str           # "long", "short", "neutral"
    summary: str

    whale_txs: List[WhaleTransaction] = None
    exchange_flows: Optional[ExchangeFlows] = None
    stablecoin_flow: Optional[StablecoinFlow] = None


class OnChainAnalyzer:
    """
    Analyzes on-chain data for trading signals.

    Components:
    1. Whale transactions (large movements)
    2. Exchange inflow/outflow (selling/accumulation)
    3. Stablecoin movements (buying pressure)

    Note: This requires external APIs (Whale Alert, Glassnode, etc.)
    Without API keys, returns neutral scores.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self.whale_alert_key = WHALE_ALERT_API_KEY

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_whale_transactions(
        self,
        symbol: str = "btc",
        min_usd: float = None
    ) -> List[WhaleTransaction]:
        """
        Fetch whale transactions from Whale Alert API.

        Note: Requires WHALE_ALERT_API_KEY env var.
        """
        min_usd = min_usd or ONCHAIN_PARAMS.whale_min_usd

        if not self.whale_alert_key:
            logger.debug("Whale Alert API key not configured")
            return []

        session = await self._get_session()
        url = "https://api.whale-alert.io/v1/transactions"

        # Get last 24 hours
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (24 * 3600)

        params = {
            "api_key": self.whale_alert_key,
            "min_value": int(min_usd),
            "start": start_time,
            "end": end_time,
            "currency": symbol.lower()
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.debug(f"Whale Alert API error: {response.status}")
                    return []

                data = await response.json()
                transactions = []

                for tx in data.get("transactions", []):
                    # Determine if exchange-related
                    from_type = tx.get("from", {}).get("owner_type", "")
                    to_type = tx.get("to", {}).get("owner_type", "")

                    transactions.append(WhaleTransaction(
                        amount_usd=float(tx.get("amount_usd", 0)),
                        from_exchange=from_type == "exchange",
                        to_exchange=to_type == "exchange",
                        symbol=tx.get("symbol", "").upper(),
                        timestamp=datetime.fromtimestamp(tx.get("timestamp", 0))
                    ))

                return transactions

        except Exception as e:
            logger.error(f"Error fetching whale transactions: {e}")
            return []

    def analyze_whale_flow(self, transactions: List[WhaleTransaction]) -> Dict:
        """
        Analyze whale transaction flow.

        Returns:
            {
                'to_exchange_usd': float,   # Bearish signal
                'from_exchange_usd': float, # Bullish signal
                'score': float,             # 0-100
                'summary': str
            }
        """
        to_exchange = sum(
            tx.amount_usd for tx in transactions if tx.to_exchange
        )
        from_exchange = sum(
            tx.amount_usd for tx in transactions if tx.from_exchange
        )

        total = to_exchange + from_exchange

        if total == 0:
            return {
                'to_exchange_usd': 0,
                'from_exchange_usd': 0,
                'score': 50,
                'summary': "No whale activity"
            }

        # Score: More outflow = bullish
        outflow_ratio = from_exchange / total if total > 0 else 0.5

        if outflow_ratio > 0.6:
            score = 60 + (outflow_ratio - 0.6) * 100
            summary = f"${from_exchange/1e6:.1f}M withdrawn"
        elif outflow_ratio < 0.4:
            score = 40 - (0.4 - outflow_ratio) * 100
            summary = f"${to_exchange/1e6:.1f}M deposited"
        else:
            score = 50
            summary = "Neutral whale flow"

        return {
            'to_exchange_usd': to_exchange,
            'from_exchange_usd': from_exchange,
            'score': min(100, max(0, score)),
            'summary': summary
        }

    async def fetch_exchange_flows(self, symbol: str = "BTC") -> Optional[ExchangeFlows]:
        """
        Fetch exchange inflow/outflow data.

        Note: Would typically use Glassnode or similar API.
        Returns mock data without API key.
        """
        # Without API, return neutral
        return ExchangeFlows(
            inflow_usd=0,
            outflow_usd=0,
            net_flow=0,
            flow_bias="neutral"
        )

    async def fetch_stablecoin_flows(self) -> Optional[StablecoinFlow]:
        """
        Fetch stablecoin inflow to exchanges.

        Large stablecoin inflows = buying pressure (bullish).
        """
        # Without API, return neutral
        return StablecoinFlow(
            inflow_usd=0,
            is_bullish=False
        )

    async def analyze(self, symbol: str = "BTC") -> OnChainScore:
        """
        Analyze all on-chain data and return combined score.
        """
        # Map common trading pairs to chain symbols
        chain_symbol = symbol.replace("USDT", "").replace("BUSD", "").upper()
        if chain_symbol in ["BTC", "ETH", "SOL", "DOGE", "XRP"]:
            pass  # Valid
        else:
            chain_symbol = "BTC"  # Default to BTC for unknown

        # Fetch all data
        whale_txs, exchange_flows, stable_flows = await asyncio.gather(
            self.fetch_whale_transactions(chain_symbol),
            self.fetch_exchange_flows(chain_symbol),
            self.fetch_stablecoin_flows(),
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(whale_txs, Exception):
            whale_txs = []
        if isinstance(exchange_flows, Exception):
            exchange_flows = None
        if isinstance(stable_flows, Exception):
            stable_flows = None

        # Calculate component scores
        scores = []
        summaries = []

        # 1. Whale flow (50% of on-chain)
        whale_analysis = self.analyze_whale_flow(whale_txs)
        scores.append(("whale", whale_analysis['score'], 0.50))
        if whale_analysis['summary'] != "No whale activity":
            summaries.append(whale_analysis['summary'])

        # 2. Exchange flows (30% of on-chain)
        if exchange_flows and exchange_flows.flow_bias != "neutral":
            if exchange_flows.flow_bias == "bullish":
                ef_score = 70
                summaries.append("Exchange outflow")
            else:
                ef_score = 30
                summaries.append("Exchange inflow")
        else:
            ef_score = 50
        scores.append(("exchange", ef_score, 0.30))

        # 3. Stablecoin flows (20% of on-chain)
        if stable_flows and stable_flows.is_bullish:
            sf_score = 70
            summaries.append(f"${stable_flows.inflow_usd/1e6:.0f}M stables in")
        else:
            sf_score = 50
        scores.append(("stablecoin", sf_score, 0.20))

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
        summary = " | ".join(summaries[:2]) if summaries else "No on-chain data"

        return OnChainScore(
            total_score=round(total_score, 1),
            direction=direction,
            summary=summary,
            whale_txs=whale_txs,
            exchange_flows=exchange_flows,
            stablecoin_flow=stable_flows
        )


# Singleton
_analyzer: Optional[OnChainAnalyzer] = None


def get_onchain_analyzer() -> OnChainAnalyzer:
    """Get the on-chain analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OnChainAnalyzer()
    return _analyzer


async def analyze_onchain(symbol: str = "BTC") -> OnChainScore:
    """Convenience function to analyze on-chain data."""
    analyzer = get_onchain_analyzer()
    return await analyzer.analyze(symbol)
