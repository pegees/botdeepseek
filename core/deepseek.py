"""DeepSeek API client with retry logic and timeouts."""
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any

import aiohttp

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)
from core.exceptions import DeepSeekAPIError, RateLimitError

logger = logging.getLogger(__name__)

# Prompt for single signal analysis
SINGLE_SIGNAL_PROMPT = """You are a professional crypto scalp trader. Analyze the following market data and find the BEST scalp setup.

TIMEFRAME: {timeframe}
STYLE: {style} (quick entries, tight stops)

MARKET DATA:
{market_data}

ANALYSIS CRITERIA:
1. Look for liquidity sweeps (price took out recent high/low then reversed)
2. Volume confirmation (volume_ratio > 1.2 suggests interest)
3. Clean support/resistance levels
4. Funding rate extremes can signal reversals
5. Avoid choppy/ranging pairs with no clear direction

RESPOND IN THIS EXACT FORMAT:
PAIR: [symbol]
DIRECTION: [LONG or SHORT]
ENTRY: [price]
STOP LOSS: [price]
TAKE PROFIT: [price]
RISK/REWARD: [ratio like 1:2]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [2-3 sentences max explaining the setup]

If no good setup exists, start your response with "NO CLEAR SETUP" and explain why.

Pick the single best opportunity right now."""

# Prompt for multiple signal analysis
# Prompt with indicator data for enhanced analysis
INDICATOR_ENHANCED_PROMPT = """You are a professional crypto scalp trader. Analyze the pre-calculated indicator data below and provide the BEST trading signal.

TIMEFRAME: {timeframe}
STYLE: {style} (quick entries, tight stops)

PRE-CALCULATED INDICATOR DATA:
{indicator_data}

RAW MARKET DATA:
{market_data}

The indicators have already been calculated:
- RSI: Overbought (>70) or Oversold (<30) signals potential reversal
- MACD: Crossovers and histogram shifts indicate momentum change
- EMA: Fast/slow crossovers show trend direction
- Market Structure: HH/HL = uptrend, LL/LH = downtrend, BOS = trend change
- Volume: Spikes (>2x average) confirm moves
- CVD: Divergence between price and volume delta suggests reversal
- Liquidity Sweep: Wick beyond key level then reversal = manipulation
- FVG: Fair value gaps often get filled

CONFLUENCE SUMMARY:
{confluence_summary}

Based on the indicator confluence, provide a trading signal:

RESPOND IN THIS EXACT FORMAT:
PAIR: [symbol]
DIRECTION: [LONG or SHORT]
ENTRY: [price]
STOP LOSS: [price]
TAKE PROFIT: [price]
RISK/REWARD: [ratio like 1:2]
CONFIDENCE: [LOW/MEDIUM/HIGH based on indicator confluence]
REASONING: [2-3 sentences explaining which indicators align and why]

If indicators are conflicting or no clear setup exists, respond with "NO CLEAR SETUP" and explain."""

MULTI_SIGNAL_PROMPT = """You are a professional crypto scalp trader. Analyze the following market data and find the TOP {count} scalp setups.

TIMEFRAME: {timeframe}
STYLE: {style} (quick entries, tight stops)

MARKET DATA:
{market_data}

ANALYSIS CRITERIA:
1. Look for liquidity sweeps (price took out recent high/low then reversed)
2. Volume confirmation (volume_ratio > 1.2 suggests interest)
3. Clean support/resistance levels
4. Funding rate extremes can signal reversals
5. Avoid choppy/ranging pairs with no clear direction

RESPOND WITH EXACTLY {count} SETUPS, ranked by confidence. Use this format for EACH:

SETUP 1:
PAIR: [symbol]
DIRECTION: [LONG or SHORT]
ENTRY: [price]
STOP LOSS: [price]
TAKE PROFIT: [price]
RISK/REWARD: [ratio like 1:2]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [1-2 sentences]

SETUP 2:
...

If fewer than {count} good setups exist, provide as many as you can find."""


class DeepSeekClient:
    """Async DeepSeek API client with connection pooling and retry logic."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)

    async def __aenter__(self) -> "DeepSeekClient":
        """Create session on context entry."""
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close session on context exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, payload: Dict[str, Any]) -> str:
        """
        Make a request with exponential backoff retry logic.

        Args:
            payload: Request payload for DeepSeek API

        Returns:
            Response content string

        Raises:
            RateLimitError: If rate limit is exceeded
            DeepSeekAPIError: If request fails after all retries
        """
        if not self._session:
            raise DeepSeekAPIError("Client session not initialized. Use 'async with' context manager.")

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.post(DEEPSEEK_API_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        try:
                            return data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError) as e:
                            raise DeepSeekAPIError(f"Unexpected response structure: {e}")

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After", "60")
                        raise RateLimitError(
                            f"DeepSeek rate limit exceeded. Retry after {retry_after}s"
                        )

                    if response.status >= 500:
                        error_text = await response.text()
                        raise DeepSeekAPIError(f"Server error {response.status}: {error_text}")

                    # Client errors (4xx except 429)
                    error_text = await response.text()
                    raise DeepSeekAPIError(f"API error {response.status}: {error_text}")

            except asyncio.TimeoutError:
                last_error = DeepSeekAPIError(f"Request timed out after {DEEPSEEK_TIMEOUT}s")
                logger.warning(f"DeepSeek request timeout (attempt {attempt + 1}/{MAX_RETRIES})")

            except aiohttp.ClientError as e:
                last_error = DeepSeekAPIError(f"Connection error: {e}")
                logger.warning(f"DeepSeek connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

            except RateLimitError:
                raise  # Don't retry rate limits

            except DeepSeekAPIError as e:
                if "Server error" in str(e):
                    last_error = e
                    logger.warning(f"DeepSeek server error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                else:
                    raise  # Don't retry client errors

            # Exponential backoff before retry
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

        raise DeepSeekAPIError(f"Failed after {MAX_RETRIES} attempts: {last_error}")

    async def analyze(
        self,
        market_data: List[Dict],
        timeframe: str,
        style: str = "scalp",
    ) -> str:
        """
        Send market data to DeepSeek for single signal analysis.

        Args:
            market_data: List of market data dicts from BinanceClient
            timeframe: Trading timeframe (e.g., '15m')
            style: Trading style (default 'scalp')

        Returns:
            Analysis response string from DeepSeek
        """
        if not market_data:
            return "NO CLEAR SETUP: No market data available for analysis."

        prompt = SINGLE_SIGNAL_PROMPT.format(
            timeframe=timeframe,
            style=style,
            market_data=json.dumps(market_data, indent=2),
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional crypto trader specializing in scalping. Be concise and precise.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }

        logger.info(f"Requesting DeepSeek analysis for {len(market_data)} pairs on {timeframe}")
        response = await self._request(payload)
        logger.info("DeepSeek analysis completed")

        return response

    async def analyze_multi(
        self,
        market_data: List[Dict],
        timeframe: str,
        count: int = 3,
        style: str = "scalp",
    ) -> str:
        """
        Send market data to DeepSeek for multiple signal analysis.

        Args:
            market_data: List of market data dicts from BinanceClient
            timeframe: Trading timeframe (e.g., '15m')
            count: Number of signals to return
            style: Trading style (default 'scalp')

        Returns:
            Analysis response string with multiple setups
        """
        if not market_data:
            return "NO CLEAR SETUP: No market data available for analysis."

        prompt = MULTI_SIGNAL_PROMPT.format(
            timeframe=timeframe,
            style=style,
            count=count,
            market_data=json.dumps(market_data, indent=2),
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional crypto trader specializing in scalping. Be concise and precise.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }

        logger.info(f"Requesting DeepSeek multi-analysis ({count} signals) for {len(market_data)} pairs")
        response = await self._request(payload)
        logger.info("DeepSeek multi-analysis completed")

        return response

    async def analyze_with_indicators(
        self,
        market_data: List[Dict],
        indicator_data: Dict[str, Any],
        confluence: Dict[str, int],
        timeframe: str,
        style: str = "scalp",
    ) -> str:
        """
        Send market data WITH pre-calculated indicators to DeepSeek.

        This hybrid approach gives DeepSeek the raw data PLUS indicator
        calculations so it can make a more informed decision.

        Args:
            market_data: List of market data dicts from BinanceClient
            indicator_data: Dict of indicator results from ScannerService
            confluence: Confluence counts (bullish, bearish, neutral)
            timeframe: Trading timeframe (e.g., '15m')
            style: Trading style (default 'scalp')

        Returns:
            Analysis response string from DeepSeek
        """
        if not market_data:
            return "NO CLEAR SETUP: No market data available for analysis."

        # Build confluence summary
        total = confluence.get("bullish", 0) + confluence.get("bearish", 0) + confluence.get("neutral", 0)
        confluence_summary = f"""
Bullish indicators: {confluence.get('bullish', 0)}/{total}
Bearish indicators: {confluence.get('bearish', 0)}/{total}
Neutral indicators: {confluence.get('neutral', 0)}/{total}
Overall bias: {'BULLISH' if confluence.get('bullish', 0) > confluence.get('bearish', 0) else 'BEARISH' if confluence.get('bearish', 0) > confluence.get('bullish', 0) else 'NEUTRAL'}
"""

        # Format indicator data for readability
        formatted_indicators = self._format_indicators(indicator_data)

        prompt = INDICATOR_ENHANCED_PROMPT.format(
            timeframe=timeframe,
            style=style,
            indicator_data=formatted_indicators,
            market_data=json.dumps(market_data[:3], indent=2),  # Top 3 pairs only to save tokens
            confluence_summary=confluence_summary,
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional crypto trader specializing in scalping. You receive pre-calculated technical indicators to help you make decisions. Trust the indicator calculations and focus on synthesis.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 600,
        }

        logger.info(f"Requesting enhanced DeepSeek analysis with {len(indicator_data)} indicators")
        response = await self._request(payload)
        logger.info("DeepSeek enhanced analysis completed")

        return response

    def _format_indicators(self, indicators: Dict[str, Any]) -> str:
        """Format indicator data for the prompt."""
        lines = []

        for name, data in indicators.items():
            if isinstance(data, dict):
                signal = data.get("signal", "?")
                value = data.get("value", "?")
                strength = data.get("strength", 0)

                # Format based on indicator type
                if name == "rsi":
                    lines.append(f"RSI: {value} ({signal}, strength: {strength})")
                    if data.get("metadata", {}).get("divergence"):
                        lines.append(f"  - Divergence detected: {data['metadata']['divergence']}")
                elif name == "macd":
                    meta = data.get("metadata", {})
                    lines.append(f"MACD: {signal} (strength: {strength})")
                    if meta.get("crossover"):
                        lines.append(f"  - Crossover: {meta['crossover']}")
                    if meta.get("histogram_shift"):
                        lines.append(f"  - Histogram shift: {meta['histogram_shift']}")
                elif name == "ema":
                    meta = data.get("metadata", {})
                    lines.append(f"EMA: {signal} (strength: {strength})")
                    lines.append(f"  - Spread: {meta.get('spread_pct', 0):.2f}%")
                    if meta.get("crossover"):
                        lines.append(f"  - Crossover: {meta['crossover']}")
                elif name == "market_structure":
                    meta = data.get("metadata", {})
                    lines.append(f"Market Structure: {meta.get('trend', 'unknown')} ({signal})")
                    if meta.get("bos"):
                        lines.append(f"  - Break of Structure: {meta['bos'].get('type', '?')}")
                elif name == "volume":
                    meta = data.get("metadata", {})
                    lines.append(f"Volume: {value}x average ({signal})")
                    if meta.get("is_spike"):
                        lines.append("  - Volume SPIKE detected")
                elif name == "cvd":
                    meta = data.get("metadata", {})
                    lines.append(f"CVD: {signal} (strength: {strength})")
                    if meta.get("divergence"):
                        lines.append(f"  - CVD Divergence: {meta['divergence']}")
                elif name == "liquidity_sweep":
                    meta = data.get("metadata", {})
                    if meta.get("sweep_detected"):
                        sweep = meta.get("sweep", {})
                        lines.append(f"Liquidity Sweep: {sweep.get('type', '?')} detected!")
                        lines.append(f"  - Swept level: {sweep.get('swept_level', '?')}")
                elif name == "fvg":
                    meta = data.get("metadata", {})
                    if meta.get("unfilled_bullish") or meta.get("unfilled_bearish"):
                        lines.append(f"FVG: {meta.get('unfilled_bullish', 0)} bullish, {meta.get('unfilled_bearish', 0)} bearish gaps nearby")
                elif name == "support_resistance":
                    meta = data.get("metadata", {})
                    lines.append(f"S/R Levels: {signal}")
                    if meta.get("nearest_support"):
                        lines.append(f"  - Support: {meta['nearest_support']}")
                    if meta.get("nearest_resistance"):
                        lines.append(f"  - Resistance: {meta['nearest_resistance']}")
                else:
                    lines.append(f"{name}: {value} ({signal}, strength: {strength})")

        return "\n".join(lines) if lines else "No indicator data available"
