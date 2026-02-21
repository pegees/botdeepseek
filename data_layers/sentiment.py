"""
Sentiment Data Layer
=====================
Fear & Greed Index, social volume, and market regime classification.
Weight: 10% of confluence score.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

from config import SENTIMENT_PARAMS, LUNARCRUSH_API_KEY

logger = logging.getLogger(__name__)


@dataclass
class FearGreedData:
    """Fear & Greed Index data."""
    value: int               # 0-100
    classification: str      # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    is_contrarian_buy: bool  # Extreme fear = buy signal
    is_contrarian_sell: bool # Extreme greed = sell signal


@dataclass
class SocialData:
    """Social sentiment data."""
    social_volume: float     # Relative volume (1.0 = normal)
    sentiment: float         # -1 to 1 (bearish to bullish)
    is_spike: bool           # Volume spike detected
    top_mentions: List[str]  # Top mentioned coins


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str              # "trending_up", "trending_down", "ranging"
    adx: float               # ADX value
    trend_strength: str      # "strong", "moderate", "weak"


@dataclass
class SentimentScore:
    """Combined sentiment score."""
    total_score: float       # 0-100
    direction: str           # "long", "short", "neutral"
    summary: str

    fear_greed: Optional[FearGreedData] = None
    social: Optional[SocialData] = None
    regime: Optional[MarketRegime] = None


class SentimentAnalyzer:
    """
    Analyzes sentiment data for trading signals.

    Components:
    1. Fear & Greed Index (contrarian indicator)
    2. Social volume and sentiment
    3. Market regime (trending vs ranging)
    """

    FEAR_GREED_API = "https://api.alternative.me/fng/"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self.lunarcrush_key = LUNARCRUSH_API_KEY

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_fear_greed(self) -> Optional[FearGreedData]:
        """
        Fetch Fear & Greed Index from alternative.me API.
        Free, no API key required.
        """
        session = await self._get_session()

        try:
            async with session.get(self.FEAR_GREED_API) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                fng_data = data.get("data", [{}])[0]

                value = int(fng_data.get("value", 50))
                classification = fng_data.get("value_classification", "Neutral")

                return FearGreedData(
                    value=value,
                    classification=classification,
                    is_contrarian_buy=value <= SENTIMENT_PARAMS.extreme_fear,
                    is_contrarian_sell=value >= SENTIMENT_PARAMS.extreme_greed
                )

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
            return None

    async def fetch_social_data(self, symbol: str = "BTC") -> Optional[SocialData]:
        """
        Fetch social sentiment from LunarCrush API.

        Note: Requires LUNARCRUSH_API_KEY env var.
        """
        if not self.lunarcrush_key:
            logger.debug("LunarCrush API key not configured")
            return None

        session = await self._get_session()
        url = f"https://lunarcrush.com/api3/coins/{symbol}/time-series"

        params = {
            "key": self.lunarcrush_key,
            "interval": "hour",
            "data_points": 24
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                timeseries = data.get("data", [])

                if not timeseries:
                    return None

                # Calculate relative volume
                volumes = [ts.get("social_volume", 0) for ts in timeseries]
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else 1
                current_volume = volumes[-1] if volumes else 0
                relative_volume = current_volume / avg_volume if avg_volume > 0 else 1

                # Get sentiment
                latest = timeseries[-1] if timeseries else {}
                sentiment_raw = latest.get("sentiment", 50) / 100  # Normalize to 0-1

                return SocialData(
                    social_volume=relative_volume,
                    sentiment=sentiment_raw * 2 - 1,  # Convert to -1 to 1
                    is_spike=relative_volume >= SENTIMENT_PARAMS.social_spike_threshold,
                    top_mentions=[]
                )

        except Exception as e:
            logger.error(f"Error fetching social data: {e}")
            return None

    def calculate_market_regime(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float]
    ) -> MarketRegime:
        """
        Calculate market regime using ADX.

        ADX > 25 = Trending
        ADX < 20 = Ranging
        """
        if len(closes) < 20:
            return MarketRegime(
                regime="unknown",
                adx=0,
                trend_strength="unknown"
            )

        # Simplified ADX calculation
        import numpy as np

        # Calculate True Range
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        # Smooth with 14-period EMA
        period = 14

        def ema(data, period):
            alpha = 2 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                result.append(alpha * data[i] + (1 - alpha) * result[-1])
            return result

        atr = ema(tr_list[-period*2:], period)[-1]
        plus_di = (ema(plus_dm[-period*2:], period)[-1] / atr) * 100 if atr > 0 else 0
        minus_di = (ema(minus_dm[-period*2:], period)[-1] / atr) * 100 if atr > 0 else 0

        # ADX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = (di_diff / di_sum * 100) if di_sum > 0 else 0

        # Simplified: use DX as ADX approximation
        adx = dx

        # Determine regime
        if adx >= SENTIMENT_PARAMS.trending_adx_threshold:
            if plus_di > minus_di:
                regime = "trending_up"
            else:
                regime = "trending_down"
            strength = "strong" if adx >= 35 else "moderate"
        elif adx <= SENTIMENT_PARAMS.ranging_adx_threshold:
            regime = "ranging"
            strength = "weak"
        else:
            regime = "transitioning"
            strength = "moderate"

        return MarketRegime(
            regime=regime,
            adx=adx,
            trend_strength=strength
        )

    async def analyze(
        self,
        symbol: str = "BTC",
        closes: List[float] = None,
        highs: List[float] = None,
        lows: List[float] = None
    ) -> SentimentScore:
        """
        Analyze all sentiment data and return combined score.
        """
        # Fetch external data
        fear_greed, social = await asyncio.gather(
            self.fetch_fear_greed(),
            self.fetch_social_data(symbol),
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(fear_greed, Exception):
            fear_greed = None
        if isinstance(social, Exception):
            social = None

        # Calculate regime if price data provided
        if closes and highs and lows:
            regime = self.calculate_market_regime(closes, highs, lows)
        else:
            regime = None

        # Calculate component scores
        scores = []
        summaries = []

        # 1. Fear & Greed (50% of sentiment)
        if fear_greed:
            if fear_greed.is_contrarian_buy:
                fg_score = 80
                summaries.append(f"Fear {fear_greed.value}")
            elif fear_greed.is_contrarian_sell:
                fg_score = 20
                summaries.append(f"Greed {fear_greed.value}")
            else:
                # Linear scale: Fear = bullish, Greed = bearish (contrarian)
                fg_score = 100 - fear_greed.value
                summaries.append(f"F&G {fear_greed.value}")
            scores.append(("fear_greed", fg_score, 0.50))
        else:
            scores.append(("fear_greed", 50, 0.50))

        # 2. Social sentiment (30% of sentiment)
        if social:
            # Sentiment -1 to 1 mapped to 0-100
            social_score = (social.sentiment + 1) * 50

            if social.is_spike:
                summaries.append(f"Social spike {social.social_volume:.1f}x")
            scores.append(("social", social_score, 0.30))
        else:
            scores.append(("social", 50, 0.30))

        # 3. Market regime (20% of sentiment)
        if regime:
            if regime.regime == "trending_up":
                regime_score = 70
                summaries.append(f"Trend↑ ADX {regime.adx:.0f}")
            elif regime.regime == "trending_down":
                regime_score = 30
                summaries.append(f"Trend↓ ADX {regime.adx:.0f}")
            else:
                regime_score = 50
                summaries.append(f"Range ADX {regime.adx:.0f}")
            scores.append(("regime", regime_score, 0.20))
        else:
            scores.append(("regime", 50, 0.20))

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
        summary = " | ".join(summaries[:2]) if summaries else "No sentiment data"

        return SentimentScore(
            total_score=round(total_score, 1),
            direction=direction,
            summary=summary,
            fear_greed=fear_greed,
            social=social,
            regime=regime
        )


# Singleton
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get the sentiment analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


async def analyze_sentiment(
    symbol: str = "BTC",
    closes: List[float] = None,
    highs: List[float] = None,
    lows: List[float] = None
) -> SentimentScore:
    """Convenience function to analyze sentiment."""
    analyzer = get_sentiment_analyzer()
    return await analyzer.analyze(symbol, closes, highs, lows)
