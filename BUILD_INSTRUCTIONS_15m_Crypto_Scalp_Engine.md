# BUILD INSTRUCTIONS: Data-Driven 15m Crypto Scalp Engine — Telegram Signal Bot

> **Target Audience:** AI Programmer / Developer
> **Purpose:** Build a fully automated, emotion-free, data-driven cryptocurrency scalping system specialized for the 15-minute timeframe. Every signal must be backed by quantifiable data confluence across 4 data layers. Zero guesswork. Zero emotion. Pure probability.
> **Primary Timeframe:** 15m (with 1m, 5m, 1h as confirmation)
> **Market:** All Binance USDT perpetual & spot pairs
> **Output:** Telegram bot that sends high-confidence scalp signals

---

## TABLE OF CONTENTS

1. [Project Setup & File Structure](#1-project-setup--file-structure)
2. [Market Scanner — Pair Discovery Engine](#2-market-scanner--pair-discovery-engine)
3. [Data Layer 1 — Price Action & Technical Analysis Engine](#3-data-layer-1--price-action--technical-analysis-engine)
4. [Data Layer 2 — Order Flow & Market Microstructure](#4-data-layer-2--order-flow--market-microstructure)
5. [Data Layer 3 — On-Chain Intelligence](#5-data-layer-3--on-chain-intelligence)
6. [Data Layer 4 — Sentiment & Macro Data](#6-data-layer-4--sentiment--macro-data)
7. [Confluence Scoring Engine (The Brain)](#7-confluence-scoring-engine-the-brain)
8. [Risk Management Engine (Non-Negotiable)](#8-risk-management-engine-non-negotiable)
9. [Anti-Emotion Safeguards & Circuit Breakers](#9-anti-emotion-safeguards--circuit-breakers)
10. [Telegram Bot — Signal Delivery](#10-telegram-bot--signal-delivery)
11. [Database & Logging System](#11-database--logging-system)
12. [Backtesting Module](#12-backtesting-module)
13. [Deployment & Infrastructure](#13-deployment--infrastructure)
14. [API Keys & External Services Checklist](#14-api-keys--external-services-checklist)
15. [Development Roadmap](#15-development-roadmap)

---

## 1. PROJECT SETUP & FILE STRUCTURE

### 1.1 Create This Exact Directory Structure

```
crypto-scalp-engine/
├── .env                          # API keys & config
├── .env.example                  # Template for .env
├── docker-compose.yml            # Deployment
├── Dockerfile
├── requirements.txt
├── config/
│   ├── settings.py               # Global settings & constants
│   ├── pairs_config.py           # Pair filter thresholds
│   └── strategy_params.py        # All tunable strategy parameters
├── core/
│   ├── __init__.py
│   ├── scanner.py                # Market scanner (Section 2)
│   ├── engine.py                 # Main orchestrator loop
│   └── confluence.py             # Confluence scoring (Section 7)
├── data_layers/
│   ├── __init__.py
│   ├── technical.py              # TA engine (Section 3)
│   ├── orderflow.py              # Order flow (Section 4)
│   ├── onchain.py                # On-chain data (Section 5)
│   └── sentiment.py              # Sentiment data (Section 6)
├── risk/
│   ├── __init__.py
│   ├── position_sizing.py        # Position size calculator
│   ├── risk_manager.py           # SL/TP/R:R logic
│   └── circuit_breakers.py       # Anti-emotion safeguards (Section 9)
├── telegram_bot/
│   ├── __init__.py
│   ├── bot.py                    # Main bot setup
│   ├── formatter.py              # Signal message formatting
│   ├── commands.py               # /stats, /pnl, /winrate handlers
│   └── alerts.py                 # Alert dispatcher
├── database/
│   ├── __init__.py
│   ├── models.py                 # SQLAlchemy models
│   ├── schema.sql                # Raw SQL schema
│   └── db.py                     # DB connection & helpers
├── backtesting/
│   ├── __init__.py
│   ├── data_fetcher.py           # Historical data downloader
│   ├── replay_engine.py          # Strategy replay
│   └── metrics.py                # Performance calculations
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Structured logging
│   ├── helpers.py                # Common utilities
│   └── normalizer.py             # Score normalization functions
├── tests/
│   └── ...
└── main.py                       # Entry point
```

### 1.2 Dependencies — `requirements.txt`

```
# Exchange & Market Data
ccxt==4.4.26
websockets==12.0
aiohttp==3.9.5

# Technical Analysis
pandas==2.2.2
pandas-ta==0.3.14b
numpy==1.26.4
scipy==1.13.0

# Machine Learning
scikit-learn==1.5.0
xgboost==2.0.3
joblib==1.4.2

# NLP / Sentiment
textblob==0.18.0
vaderSentiment==3.3.2
transformers==4.41.0

# Telegram
aiogram==3.7.0

# Database
sqlalchemy==2.0.30
asyncpg==0.29.0
psycopg2-binary==2.9.9
alembic==1.13.1

# On-Chain
web3==6.19.0
requests==2.32.3

# Redis (real-time cache)
redis==5.0.4
aioredis==2.0.1

# Utilities
python-dotenv==1.0.1
loguru==0.7.2
pydantic==2.7.3
schedule==1.2.2
```

### 1.3 Environment Variables — `.env.example`

```bash
# Exchange
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHANNEL_ID=
TELEGRAM_ADMIN_ID=

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/scalp_engine

# Redis
REDIS_URL=redis://localhost:6379/0

# On-Chain
ETHERSCAN_API_KEY=
WHALE_ALERT_API_KEY=

# Sentiment
TWITTER_BEARER_TOKEN=
LUNARCRUSH_API_KEY=

# Settings
MAX_CONCURRENT_POSITIONS=5
RISK_PER_TRADE_PCT=1.5
MAX_DAILY_DRAWDOWN_PCT=5.0
MIN_RR_RATIO=2.0
SIGNAL_EXPIRY_MINUTES=15
```

### 1.4 Global Config — `config/settings.py`

```python
"""
ALL tunable parameters live here. Never hardcode values in logic files.
The programmer must make every threshold configurable from this file.
"""

# Primary timeframe
PRIMARY_TIMEFRAME = "15m"
CONFIRMATION_TIMEFRAMES = ["1m", "5m", "1h"]

# Scanner thresholds
MIN_24H_VOLUME_USDT = 500_000        # Minimum $500K daily volume
MIN_LIQUIDITY_SCORE = 60              # 0-100 scale
MAX_SPREAD_PCT = 0.10                 # Maximum 0.1% spread
SCANNER_REFRESH_INTERVAL_SEC = 60     # Re-scan every 60 seconds

# Signal thresholds
HIGH_CONFIDENCE_THRESHOLD = 80        # Score 80-100 = send immediately
MEDIUM_CONFIDENCE_THRESHOLD = 65      # Score 65-79 = send with caution
SIGNAL_EXPIRY_MINUTES = 15            # Signal invalid after 15 min (1 candle)

# Risk management
RISK_PER_TRADE_PCT = 1.5              # Risk 1.5% per trade
MAX_PORTFOLIO_HEAT_PCT = 6.0          # Max 6% total exposure
MIN_RR_RATIO = 2.0                    # Reject signals below 1:2 R:R
MAX_CONCURRENT_POSITIONS = 5
ATR_SL_MULTIPLIER = 1.5              # SL = entry ± (ATR * 1.5)
ATR_PERIOD = 14

# Anti-emotion
MAX_SIGNALS_PER_HOUR = 10
REVENGE_TRADE_COOLDOWN_SEC = 900      # 15 min after SL hit
DRAWDOWN_CIRCUIT_BREAKER_PCT = 5.0    # Pause at -5% daily
WIN_STREAK_COOLDOWN_THRESHOLD = 5     # Reduce size after 5 wins
CORRELATION_BLOCK_THRESHOLD = 0.85    # Block if pair correlation > 0.85

# Confluence weights (must sum to 1.0)
WEIGHT_TECHNICAL = 0.35
WEIGHT_ORDERFLOW = 0.25
WEIGHT_ONCHAIN = 0.20
WEIGHT_SENTIMENT = 0.10
WEIGHT_BACKTEST = 0.10

# Scalp-specific settings (15m focused)
SCALP_TARGET_HOLD_TIME_CANDLES = 4    # Target: hold 1-4 candles (15-60 min)
SCALP_MAX_HOLD_TIME_CANDLES = 8       # Force exit after 8 candles (2 hours)
SCALP_MIN_VOLATILITY_ATR = 0.3       # Skip low-volatility pairs (ATR too small)
SCALP_VOLUME_SPIKE_MULTIPLIER = 2.0   # Volume must be 2x avg for entry
```

---

## 2. MARKET SCANNER — PAIR DISCOVERY ENGINE

### 2.1 Objective

Continuously scan all Binance USDT pairs (400+), filter down to 20-40 tradeable pairs based on real-time data quality metrics. This is the funnel — garbage pairs never reach the signal engine.

### 2.2 Build Logic

**File:** `core/scanner.py`

**Step 1:** On startup, fetch all USDT trading pairs from Binance via REST API.

```python
# Use ccxt to fetch all markets
exchange = ccxt.binance({"enableRateLimit": True})
markets = exchange.load_markets()
usdt_pairs = [s for s in markets if s.endswith("/USDT") and markets[s]["active"]]
# This gives you 400+ pairs
```

**Step 2:** For each pair, fetch 24h ticker data and calculate these metrics:

| Metric | How to Calculate | Threshold |
|---|---|---|
| 24h Volume (USDT) | From ticker `quoteVolume` | > $500,000 |
| Bid-Ask Spread | `(ask - bid) / ask * 100` | < 0.10% |
| Price Change 24h | From ticker | Absolute value > 0.5% (not dead) |
| 15m Candle Count | Verify data availability | Must have 200+ candles of history |
| ATR (14 period, 15m) | Calculate from OHLCV | > minimum threshold (pair is moving) |

**Step 3:** Score each pair 0-100 based on weighted metrics:

```python
def calculate_opportunity_score(volume_24h, spread, atr_pct, price_change):
    volume_score = min(100, (volume_24h / 5_000_000) * 100)  # $5M = perfect score
    spread_score = max(0, 100 - (spread / 0.10 * 100))       # Lower spread = higher score
    volatility_score = min(100, (atr_pct / 1.0) * 100)       # 1% ATR = perfect
    momentum_score = min(100, abs(price_change) / 3 * 100)    # 3% move = perfect
    
    return (volume_score * 0.35 + spread_score * 0.25 + 
            volatility_score * 0.25 + momentum_score * 0.15)
```

**Step 4:** Keep only pairs scoring above 50. Sort descending. Cache in Redis. Refresh every 60 seconds.

**Step 5:** Subscribe to Binance WebSocket kline streams for all qualifying pairs on 15m timeframe.

```python
# WebSocket stream format for multiple pairs
streams = [f"{pair.lower().replace('/', '')}@kline_15m" for pair in qualified_pairs]
ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
```

### 2.3 Output

The scanner outputs a `Dict[str, PairData]` refreshed every 60 seconds:

```python
{
    "SOL/USDT": {
        "opportunity_score": 87,
        "volume_24h": 2_340_000,
        "spread_pct": 0.02,
        "atr_15m": 0.45,
        "current_price": 178.50,
        "qualified": True
    },
    ...
}
```

### 2.4 Error Handling

- If WebSocket disconnects: auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s)
- If a pair suddenly drops below volume threshold mid-session: remove from active list, do NOT generate signals for it
- Log every pair add/remove event

---

## 3. DATA LAYER 1 — PRICE ACTION & TECHNICAL ANALYSIS ENGINE

### 3.1 Objective

For every qualified pair, calculate a comprehensive set of technical indicators on the 15m timeframe, with confirmation from 1m, 5m, and 1h. Output a normalized TA Score (0-100).

### 3.2 Build Logic

**File:** `data_layers/technical.py`

**Step 1: Data Collection**

On each new 15m candle close, collect OHLCV data:
- 15m: Last 200 candles (primary analysis)
- 5m: Last 100 candles (entry timing)
- 1m: Last 60 candles (micro-structure)
- 1h: Last 50 candles (trend context)

Store as pandas DataFrames. Update incrementally from WebSocket data — do NOT re-fetch full history on each candle.

**Step 2: Calculate These Indicators (15m primary)**

Every indicator must output a sub-score from 0 to 100, where:
- 0-30 = bearish / oversold
- 31-49 = slightly bearish
- 50 = neutral
- 51-69 = slightly bullish
- 70-100 = bullish / overbought

```python
"""
INDICATOR CALCULATIONS — use pandas_ta library
Each function returns a dict: {"value": float, "score": int (0-100), "signal": str}
"""

# 1. RSI (14 period)
# Score: RSI < 30 in uptrend = 90 (oversold bounce), RSI > 70 in downtrend = 90 (short)
# RSI 40-60 = 50 (neutral)
rsi = df.ta.rsi(length=14)

# 2. EMA Crossovers (9, 21, 50, 200)
# Score: 9 > 21 > 50 > 200 (perfect bull stack) = 95
# 9 just crossed above 21 = 80 (fresh cross)
# 9 < 21 < 50 < 200 (perfect bear stack) = 5
ema_9 = df.ta.ema(length=9)
ema_21 = df.ta.ema(length=21)
ema_50 = df.ta.ema(length=50)
ema_200 = df.ta.ema(length=200)

# 3. VWAP
# Score: Price above VWAP = bullish (60-80 depending on distance)
# Price at VWAP = potential bounce zone = 70
vwap = df.ta.vwap()

# 4. MACD (12, 26, 9)
# Score: MACD cross above signal = 75-90
# Histogram growing = momentum building = +10 bonus
macd = df.ta.macd()

# 5. Bollinger Bands (20, 2)
# Score: Price at lower band in uptrend = 85 (mean reversion buy)
# Squeeze detected (bandwidth narrowing) = alert mode (score 70)
bbands = df.ta.bbands(length=20, std=2)

# 6. ATR (14) — not for scoring, used for SL/TP calculation
atr = df.ta.atr(length=14)

# 7. OBV (On-Balance Volume)
# Score: OBV divergence from price = strong signal (80-90)
# OBV confirming price = 60
obv = df.ta.obv()

# 8. Stochastic RSI (14, 14, 3, 3)
# Score: StochRSI < 20 + crossing up = 85
stochrsi = df.ta.stochrsi()

# 9. Volume Analysis
# Score: Current volume > 2x 20-period average = 85 (spike)
# Current volume < 0.5x average = 20 (no participation, skip)
volume_sma = df["volume"].rolling(20).mean()
volume_ratio = df["volume"].iloc[-1] / volume_sma.iloc[-1]

# 10. Support/Resistance Auto-Detection
# Use pivot points from last 100 candles
# Score: Price within 0.3% of support in uptrend = 90 (bounce play)
# Price within 0.3% of resistance in downtrend = 90 (rejection play)
```

**Step 3: Pattern Detection (15m candles)**

Implement these pattern detectors. Each returns a boolean + confidence score:

```python
"""
SCALP-SPECIFIC PATTERNS FOR 15M TIMEFRAME
Each returns: {"detected": bool, "type": str, "confidence": 0-100, "direction": "long"/"short"}
"""

# 1. Breakout Detection
# Logic: Price closes above resistance with volume > 2x avg
# Confidence: Higher if multiple timeframes confirm

# 2. Fakeout / Liquidity Grab Detection
# Logic: Wick pierces S/R level but body closes back inside range
# This is a REVERSAL signal — enter opposite direction
# Confidence: Higher if volume spike on wick + quick rejection

# 3. Fair Value Gap (FVG)
# Logic: Gap between candle[i-2] high and candle[i] low (bullish)
# or gap between candle[i-2] low and candle[i] high (bearish)
# Score: FVG being filled from above/below = high probability entry

# 4. EMA Bounce
# Logic: Price pulls back to 21 EMA, holds (wick touches but body stays above)
# Combined with: trend is bullish on 1h, volume increasing
# This is the bread-and-butter 15m scalp setup

# 5. VWAP Reclaim
# Logic: Price dips below VWAP, then closes back above with volume
# Strong intraday bullish signal for scalps

# 6. Momentum Ignition
# Logic: 3+ consecutive bullish candles with increasing volume
# Each candle has higher high AND higher low
# Signals strong directional momentum beginning
```

**Step 4: Multi-Timeframe Confluence Check**

```python
def check_mtf_confluence(pair, primary_signal_direction):
    """
    Check if higher timeframes agree with the 15m signal.
    This is a FILTER, not a signal generator.
    
    Returns: confluence_multiplier (0.5 to 1.5)
    """
    trend_1h = get_trend(pair, "1h")   # EMA stack direction
    trend_5m = get_trend(pair, "5m")   # Entry timing
    momentum_1m = get_momentum(pair, "1m")  # Micro confirmation
    
    # All timeframes agree = 1.5x multiplier
    # Primary + 1h agree = 1.2x
    # Primary alone = 1.0x (baseline)
    # 1h disagrees = 0.7x (caution)
    # 1h strongly disagrees = 0.5x (likely reject signal)
```

**Step 5: Combine Into TA Score**

```python
def calculate_ta_score(pair):
    """
    Combine all indicator sub-scores into one TA Score (0-100).
    """
    weights = {
        "rsi": 0.12,
        "ema_structure": 0.15,
        "vwap": 0.10,
        "macd": 0.10,
        "bollinger": 0.08,
        "obv": 0.08,
        "stochrsi": 0.07,
        "volume": 0.15,          # Volume is king for scalping
        "support_resistance": 0.10,
        "pattern": 0.05
    }
    
    raw_score = sum(scores[indicator] * weight for indicator, weight in weights.items())
    
    # Apply MTF confluence multiplier
    mtf_multiplier = check_mtf_confluence(pair, signal_direction)
    final_score = min(100, raw_score * mtf_multiplier)
    
    return {
        "ta_score": round(final_score),
        "direction": "long" if final_score > 55 else "short" if final_score < 45 else "neutral",
        "breakdown": scores,  # Individual indicator scores for Telegram message
        "key_reasons": extract_top_3_reasons(scores)  # Human-readable reasons
    }
```

### 3.3 Output Format

```python
{
    "pair": "SOL/USDT",
    "ta_score": 87,
    "direction": "long",
    "key_reasons": [
        "RSI oversold bounce at 28 (score: 88)",
        "Price touching 21 EMA support (score: 91)",
        "Volume spike 2.8x average (score: 90)"
    ],
    "entry_price": 178.35,
    "atr_15m": 0.82,
    "mtf_confluence": 1.3
}
```

### 3.4 Performance Requirement

The entire TA calculation for one pair must complete in under 50ms. For 30 pairs, the full cycle must complete within 2 seconds. Use numpy vectorized operations, never iterate row-by-row.

---

## 4. DATA LAYER 2 — ORDER FLOW & MARKET MICROSTRUCTURE

### 4.1 Objective

Analyze real-time order book dynamics and trade flow to detect smart money activity. This layer answers: "Is real money backing this move?" Output an Order Flow Score (0-100).

### 4.2 Build Logic

**File:** `data_layers/orderflow.py`

**Step 1: Order Book Depth Analysis**

Subscribe to Binance WebSocket depth stream for all qualified pairs.

```python
# Partial book depth (top 20 levels, updated every 100ms)
stream = f"{symbol.lower()}@depth20@100ms"
```

Calculate these metrics from the order book:

```python
def analyze_order_book(bids, asks):
    """
    bids/asks = list of [price, quantity] from WebSocket
    """
    
    # 1. Bid-Ask Imbalance Ratio
    # Sum of bid volume in top 10 levels vs ask volume in top 10 levels
    bid_volume = sum(qty for price, qty in bids[:10])
    ask_volume = sum(qty for price, qty in asks[:10])
    imbalance_ratio = bid_volume / (bid_volume + ask_volume)
    # > 0.6 = buying pressure (score 70-90)
    # < 0.4 = selling pressure (score 70-90 for shorts)
    # 0.45-0.55 = neutral (score 40-50)
    
    # 2. Wall Detection
    # Find abnormally large orders (> 5x average level size)
    avg_level_size = (bid_volume + ask_volume) / 20
    bid_walls = [(p, q) for p, q in bids if q > avg_level_size * 5]
    ask_walls = [(p, q) for p, q in asks if q > avg_level_size * 5]
    # Large bid wall near price = support (bullish score +15)
    # Large ask wall near price = resistance (bearish score +15)
    
    # 3. Thin Orderbook Detection
    # If total depth is abnormally low = potential for sharp move
    total_depth = bid_volume + ask_volume
    # Low depth + our signal direction = score boost (easy to move)
    
    # 4. Spoofing Detection
    # Track if large orders appear and disappear within 5 seconds
    # If detected: reduce confidence (score -20), market is being manipulated
```

**Step 2: Aggressor Trade Flow**

Subscribe to the trade stream to detect who's buying/selling aggressively.

```python
# Real-time trade stream
stream = f"{symbol.lower()}@aggTrade"
```

```python
def analyze_trade_flow(trades_last_5min):
    """
    Aggregate recent trades to detect buying/selling pressure.
    Binance aggTrade tells you if the buyer was the maker or taker.
    """
    
    # 1. Buy vs Sell Aggressor Volume
    buy_volume = sum(t["quantity"] for t in trades if t["is_buyer_maker"] == False)
    sell_volume = sum(t["quantity"] for t in trades if t["is_buyer_maker"] == True)
    buy_ratio = buy_volume / (buy_volume + sell_volume)
    # > 0.6 = aggressive buying (score 70-85)
    # > 0.75 = very aggressive buying (score 85-95)
    
    # 2. Large Trade Detection
    # Trades > 3x the average trade size in last 5 min
    avg_trade_size = total_volume / len(trades)
    large_buys = [t for t in trades if not t["is_buyer_maker"] and t["quantity"] > avg_trade_size * 3]
    large_sells = [t for t in trades if t["is_buyer_maker"] and t["quantity"] > avg_trade_size * 3]
    # Many large buys = institutional buying = score 80-95
    
    # 3. Trade Velocity
    # Number of trades per second increasing = momentum building
    # Sudden burst of trades = potential breakout/breakdown imminent
```

**Step 3: Funding Rate & Open Interest (Futures)**

```python
def analyze_futures_data(pair):
    """
    Fetch funding rate and open interest from Binance Futures API.
    Updated every 8 hours for funding, real-time for OI.
    """
    
    # 1. Funding Rate
    # Positive + high = longs paying shorts = crowded long (contrarian short signal)
    # Negative + extreme = shorts paying longs = crowded short (contrarian long signal)
    # Near zero = neutral
    funding = exchange.fetch_funding_rate(pair)
    
    # Score logic:
    # funding > +0.05% = score 30 for longs (crowded), score 80 for shorts
    # funding < -0.05% = score 80 for longs, score 30 for shorts
    # funding between -0.01% to +0.01% = score 50 (neutral)
    
    # 2. Open Interest Changes
    # OI increasing + price increasing = new longs entering = trend confirmation
    # OI increasing + price decreasing = new shorts entering = bearish
    # OI decreasing + price moving = positions closing = trend weakening
    
    # 3. Liquidation Levels (estimated)
    # Calculate where leveraged positions (10x, 25x, 50x) would get liquidated
    # Price approaching liquidation cluster = potential for cascade move
    # This is a volatility/opportunity signal, score 70-90 if near liquidation zone
```

**Step 4: Combine Into Order Flow Score**

```python
def calculate_orderflow_score(pair, signal_direction):
    weights = {
        "book_imbalance": 0.25,
        "trade_flow": 0.30,     # Actual executed trades > book orders
        "large_orders": 0.15,
        "funding_rate": 0.15,
        "open_interest": 0.10,
        "spoof_penalty": 0.05   # Negative score if spoofing detected
    }
    
    # Flip scores based on signal_direction (long vs short)
    # e.g., heavy selling = low score for longs, high score for shorts
    
    return {
        "orderflow_score": round(final_score),
        "key_reasons": ["Aggressive buying 73% of volume", "Bid wall at $178.00"],
        "data_freshness": "2 seconds ago"
    }
```

### 4.3 Performance Requirement

Order flow data must be processed within 200ms of receiving WebSocket update. Use asyncio with dedicated event handlers per pair. Buffer trade data in a deque with max 5-minute window.

---

## 5. DATA LAYER 3 — ON-CHAIN INTELLIGENCE

### 5.1 Objective

Track whale movements, exchange flows, and smart money behavior to detect institutional activity before it shows up in price. Output an On-Chain Score (0-100).

### 5.2 Build Logic

**File:** `data_layers/onchain.py`

**Step 1: Whale Movement Tracking**

```python
def track_whale_movements():
    """
    Monitor large transfers to/from exchanges.
    
    Data Sources:
    - Whale Alert API (https://whale-alert.io) — real-time alerts for transfers > $500K
    - Etherscan API — for ERC-20 token transfers
    - Blockchain.com API — for BTC transfers
    
    Implementation:
    1. Poll Whale Alert API every 30 seconds
    2. Filter for relevant tokens (match against qualified pair list)
    3. Classify transfer type:
       - Wallet → Exchange = potential sell pressure (bearish)
       - Exchange → Wallet = accumulation (bullish)
       - Exchange → Exchange = neutral/arbitrage
       - Wallet → Wallet = OTC/neutral
    """
    
    # Scoring logic:
    # Large deposit to exchange ($1M+) in last 30 min = bearish score 25 for longs
    # Large withdrawal from exchange ($1M+) in last 30 min = bullish score 80 for longs
    # Multiple withdrawals in sequence = strong accumulation = score 90
    # No significant movements = neutral score 50
```

**Step 2: Exchange Inflow/Outflow Ratio**

```python
def calculate_exchange_flow(token):
    """
    Track net flow of tokens into/out of all major exchanges.
    
    Data Source Options:
    - CryptoQuant API (best, paid)
    - Glassnode API (comprehensive, paid)
    - Build custom using Etherscan API (free but limited)
    
    Implementation:
    1. Track known exchange wallet addresses
    2. Sum all inflows (deposits) and outflows (withdrawals) per hour
    3. Calculate net flow = outflows - inflows
    """
    
    # Scoring:
    # Net outflow (more leaving exchanges) = bullish, score 70-90
    # Net inflow (more entering exchanges) = bearish, score 20-40
    # Neutral flow = score 50
    # Extreme net outflow (>2 std dev) = very bullish, score 90-100
```

**Step 3: Stablecoin Flow Monitoring**

```python
def monitor_stablecoin_flow():
    """
    Track USDT/USDC movements to exchanges = buying power arriving.
    
    Logic:
    - Large stablecoin deposit to exchange = someone is about to buy = bullish
    - Large stablecoin withdrawal = buying power leaving = bearish
    
    Implementation:
    1. Monitor USDT (TRC-20 and ERC-20) and USDC transfer events
    2. Filter transfers > $500K
    3. Identify if destination is an exchange hot wallet
    """
    
    # Scoring:
    # Large stablecoin inflow to exchanges = score 75-90 (buying power incoming)
    # Large stablecoin outflow = score 30-40
    # This is a LEADING indicator — often precedes price moves by 15-60 min
```

**Step 4: Smart Money / Top Wallet Tracking**

```python
def track_smart_money():
    """
    Monitor wallets that historically make profitable trades.
    
    Implementation:
    1. Maintain a list of known profitable wallets (from DEX tracking)
    2. When a smart wallet buys/sells a token, record it
    3. If multiple smart wallets buy the same token within 1 hour = strong signal
    
    Data Sources:
    - Nansen API (paid, best for wallet labels)
    - Arkham Intelligence API
    - DEXScreener API (free, good for DEX activity)
    """
    
    # Scoring:
    # 3+ smart wallets buying same token in 1h = score 90
    # 1-2 smart wallets = score 70
    # Smart wallets selling = score 20-30
    # No activity = score 50
```

**Step 5: Combine Into On-Chain Score**

```python
def calculate_onchain_score(pair, signal_direction):
    weights = {
        "whale_movements": 0.30,
        "exchange_flow": 0.30,
        "stablecoin_flow": 0.20,
        "smart_money": 0.20
    }
    
    # IMPORTANT: On-chain data is slower than price action.
    # Data freshness matters:
    # < 5 min old = full weight
    # 5-15 min old = 0.7x weight
    # 15-30 min old = 0.4x weight
    # > 30 min old = 0.2x weight (stale, barely useful)
    
    return {
        "onchain_score": round(final_score),
        "key_reasons": ["$2.3M SOL withdrawn from Binance 8 min ago"],
        "data_freshness": "8 minutes ago",
        "freshness_penalty": 0.7
    }
```

### 5.3 Important Notes for Programmer

- On-chain data has latency (5-30 min). NEVER treat it as real-time.
- Apply freshness decay to scores (see above).
- Cache on-chain data in Redis with TTL of 30 minutes.
- Rate limit API calls — most free tiers allow 5-10 req/min.
- For tokens only on BSC/Solana, use chain-specific APIs (BscScan, Solana RPC).

---

## 6. DATA LAYER 4 — SENTIMENT & MACRO DATA

### 6.1 Objective

Measure market sentiment and macro conditions to use as a contrarian/confirmation filter. Output a Sentiment Score (0-100).

### 6.2 Build Logic

**File:** `data_layers/sentiment.py`

**Step 1: Fear & Greed Index**

```python
def get_fear_greed():
    """
    Source: https://api.alternative.me/fng/
    Updated every 24 hours.
    
    Use as CONTRARIAN indicator for scalping:
    - Extreme Fear (0-25) = contrarian BUY bias (score 75-90 for longs)
    - Fear (25-45) = slight buy bias (score 60-70 for longs)
    - Neutral (45-55) = neutral (score 50)
    - Greed (55-75) = slight sell bias (score 40 for longs)
    - Extreme Greed (75-100) = contrarian SELL bias (score 15-25 for longs)
    """
    response = requests.get("https://api.alternative.me/fng/?limit=1")
    fng_value = int(response.json()["data"][0]["value"])
    
    # Contrarian scoring (flip the value)
    contrarian_score = 100 - fng_value
```

**Step 2: Social Volume & Mention Frequency**

```python
def analyze_social_sentiment(token):
    """
    Track mention frequency and sentiment on social platforms.
    
    Data Sources (in order of preference):
    - LunarCrush API — aggregates Twitter, Reddit, YouTube, news
    - Twitter/X API — direct mention tracking
    - Reddit API — r/CryptoCurrency, r/Bitcoin, token-specific subs
    
    Metrics to track:
    1. Mention frequency (mentions per hour vs 7-day avg)
    2. Sentiment polarity (positive vs negative mentions)
    3. Influence-weighted mentions (KOLs vs bots)
    """
    
    # Scoring for SCALPING (different from swing):
    # Sudden spike in mentions (>3x avg) + positive sentiment = score 75
    #   BUT if ALREADY at price high, this could be distribution = caution
    # Sudden spike + negative sentiment = potential capitulation = contrarian score 80
    # Gradually increasing mentions = growing interest = score 65
    # No unusual activity = neutral score 50
    # Viral negative news = score 20 for longs, 85 for shorts
    
    # USE VADER for quick sentiment analysis on scraped text
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment["compound"]  # -1 to 1
```

**Step 3: BTC Dominance & Market Regime**

```python
def analyze_market_regime():
    """
    Determine the overall market condition to contextualize scalp signals.
    
    Metrics:
    1. BTC Dominance trend (rising = risk-off, alts weak; falling = risk-on, alts strong)
    2. Total crypto market cap trend
    3. BTC price trend on 1h/4h (rising tide lifts all boats)
    """
    
    # Regime classification:
    REGIMES = {
        "RISK_ON": {  # BTC dom falling, BTC rising, total mcap rising
            "alt_long_bias": 80,
            "alt_short_bias": 30,
            "description": "Altcoin season — favor long signals on alts"
        },
        "BTC_SEASON": {  # BTC dom rising, BTC rising
            "alt_long_bias": 50,  # Alts may lag
            "alt_short_bias": 50,
            "description": "BTC absorbing capital — be selective with alt longs"
        },
        "RISK_OFF": {  # BTC falling, total mcap falling
            "alt_long_bias": 25,  # Danger zone for longs
            "alt_short_bias": 80,
            "description": "Market-wide selloff — favor shorts or sit out"
        },
        "CHOPPY": {  # No clear trend
            "alt_long_bias": 45,
            "alt_short_bias": 45,
            "description": "No clear regime — reduce position size, tighter stops"
        }
    }
```

**Step 4: Macro Correlation Tracker**

```python
def check_macro_correlation():
    """
    Check correlation with traditional markets.
    Relevant for scalping because sudden macro moves cause crypto cascades.
    
    Track:
    1. S&P 500 futures (ES) — if US markets dumping, crypto may follow
    2. DXY (Dollar Index) — strong dollar = bearish crypto
    3. US 10Y yield — sudden spikes = risk-off
    
    Implementation:
    - Use free Yahoo Finance API or Alpha Vantage
    - Check every 15 minutes (not real-time needed)
    - Flag DANGER if S&P futures drop > 1% in last hour
    - Flag DANGER if DXY spikes > 0.5% in last hour
    """
    
    # Scoring:
    # Macro calm (no significant moves) = neutral score 50
    # Macro positive (S&P up, DXY down) = bullish bias score 65-75
    # Macro negative (S&P down, DXY up) = bearish bias score 25-35
    # Macro extreme event = PAUSE all signals, send alert to Telegram
```

**Step 5: Combine Into Sentiment Score**

```python
def calculate_sentiment_score(pair, signal_direction):
    weights = {
        "fear_greed": 0.20,
        "social_sentiment": 0.30,
        "market_regime": 0.30,
        "macro_correlation": 0.20
    }
    
    return {
        "sentiment_score": round(final_score),
        "market_regime": "RISK_ON",
        "key_reasons": ["Fear & Greed at 22 (Extreme Fear) — contrarian buy"],
        "macro_warning": None  # or "S&P futures -1.2% in last hour"
    }
```

---

## 7. CONFLUENCE SCORING ENGINE (THE BRAIN)

### 7.1 Objective

Combine all 4 data layer scores into a single Confluence Score. This is the final decision maker. Only signals passing the threshold get sent to Telegram.

### 7.2 Build Logic

**File:** `core/confluence.py`

```python
def calculate_confluence_score(pair: str) -> dict:
    """
    THE CORE DECISION ENGINE.
    
    This function is called every time a new 15m candle closes.
    It orchestrates all data layers and produces the final signal.
    """
    
    # Step 1: Get scores from all data layers
    ta = technical_engine.calculate_ta_score(pair)
    of = orderflow_engine.calculate_orderflow_score(pair, ta["direction"])
    oc = onchain_engine.calculate_onchain_score(pair, ta["direction"])
    st = sentiment_engine.calculate_sentiment_score(pair, ta["direction"])
    bt = backtesting_engine.get_pattern_confidence(pair, ta["direction"])
    
    # Step 2: Apply configured weights
    raw_score = (
        ta["ta_score"] * WEIGHT_TECHNICAL +          # 0.35
        of["orderflow_score"] * WEIGHT_ORDERFLOW +    # 0.25
        oc["onchain_score"] * WEIGHT_ONCHAIN +        # 0.20
        st["sentiment_score"] * WEIGHT_SENTIMENT +    # 0.10
        bt["backtest_score"] * WEIGHT_BACKTEST         # 0.10
    )
    
    # Step 3: Apply data freshness penalties
    # On-chain data decays if stale
    freshness_adjusted = apply_freshness_penalties(raw_score, oc, st)
    
    # Step 4: Apply market regime modifier
    regime = st["market_regime"]
    if regime == "RISK_OFF" and ta["direction"] == "long":
        freshness_adjusted *= 0.7  # Penalize longs in risk-off
    elif regime == "RISK_ON" and ta["direction"] == "long":
        freshness_adjusted *= 1.1  # Boost longs in risk-on
    
    final_score = min(100, round(freshness_adjusted))
    
    # Step 5: Signal decision
    if final_score >= HIGH_CONFIDENCE_THRESHOLD:  # 80
        signal_tier = "HIGH"
        action = "SEND_IMMEDIATELY"
    elif final_score >= MEDIUM_CONFIDENCE_THRESHOLD:  # 65
        signal_tier = "MEDIUM"
        action = "SEND_WITH_CAUTION"
    else:
        signal_tier = "NONE"
        action = "DISCARD"
        return {"action": "DISCARD", "score": final_score}
    
    # Step 6: Calculate risk parameters
    risk = risk_manager.calculate_trade_params(
        pair=pair,
        direction=ta["direction"],
        entry_price=ta["entry_price"],
        atr=ta["atr_15m"],
        score=final_score
    )
    
    # Step 7: Run through circuit breakers
    if circuit_breakers.should_block(pair, ta["direction"]):
        return {"action": "BLOCKED", "reason": circuit_breakers.block_reason}
    
    # Step 8: Compile final signal
    signal = {
        "action": action,
        "pair": pair,
        "direction": ta["direction"],
        "confluence_score": final_score,
        "signal_tier": signal_tier,
        "entry_price": ta["entry_price"],
        "stop_loss": risk["stop_loss"],
        "take_profit_1": risk["tp1"],
        "take_profit_2": risk["tp2"],
        "take_profit_3": risk["tp3"],
        "rr_ratio": risk["rr_ratio"],
        "position_size_pct": risk["position_size_pct"],
        "data_breakdown": {
            "technical": {"score": ta["ta_score"], "reasons": ta["key_reasons"]},
            "orderflow": {"score": of["orderflow_score"], "reasons": of["key_reasons"]},
            "onchain": {"score": oc["onchain_score"], "reasons": oc["key_reasons"]},
            "sentiment": {"score": st["sentiment_score"], "reasons": st["key_reasons"]},
            "backtest": {"score": bt["backtest_score"], "win_rate": bt["historical_winrate"]}
        },
        "market_regime": regime,
        "timestamp": datetime.utcnow(),
        "valid_until": datetime.utcnow() + timedelta(minutes=SIGNAL_EXPIRY_MINUTES)
    }
    
    # Step 9: Log to database
    database.log_signal(signal)
    
    # Step 10: Send to Telegram
    telegram_bot.send_signal(signal)
    
    return signal
```

### 7.3 Main Engine Loop

**File:** `core/engine.py`

```python
async def main_loop():
    """
    The main orchestration loop.
    
    Trigger: Every time a 15m candle closes on ANY qualified pair.
    
    Flow:
    1. Scanner maintains qualified pairs list (refreshed every 60s)
    2. WebSocket receives candle close event
    3. For each pair with closed candle → run confluence engine
    4. If signal generated → risk check → circuit breaker check → send to Telegram
    """
    
    # Initialize all components
    scanner = MarketScanner()
    ta_engine = TechnicalEngine()
    of_engine = OrderFlowEngine()
    oc_engine = OnChainEngine()
    st_engine = SentimentEngine()
    risk_mgr = RiskManager()
    breakers = CircuitBreakers()
    bot = TelegramBot()
    
    # Start background tasks
    asyncio.create_task(scanner.run())           # Pair scanning loop
    asyncio.create_task(oc_engine.run())          # On-chain polling
    asyncio.create_task(st_engine.run())          # Sentiment polling
    asyncio.create_task(breakers.monitor())       # Circuit breaker monitoring
    
    # Main signal generation loop
    async for candle_close_event in ta_engine.stream_candle_closes():
        pair = candle_close_event["pair"]
        
        if pair not in scanner.qualified_pairs:
            continue
        
        try:
            signal = calculate_confluence_score(pair)
            if signal["action"] in ["SEND_IMMEDIATELY", "SEND_WITH_CAUTION"]:
                logger.info(f"Signal generated: {pair} {signal['direction']} score={signal['confluence_score']}")
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
            continue
```

---

## 8. RISK MANAGEMENT ENGINE (NON-NEGOTIABLE)

### 8.1 Objective

Calculate exact position sizing, stop loss, and take profit levels for every signal. These rules are ABSOLUTE — no signal bypasses risk management.

### 8.2 Build Logic

**File:** `risk/risk_manager.py`

```python
def calculate_trade_params(pair, direction, entry_price, atr, score):
    """
    Every parameter is calculated from DATA, not feelings.
    
    Inputs:
    - pair: trading pair
    - direction: "long" or "short"
    - entry_price: suggested entry
    - atr: Average True Range (14 period, 15m)
    - score: confluence score (affects position size)
    
    Returns: Complete trade parameters
    """
    
    # ============================================
    # STOP LOSS — Based on ATR, never arbitrary %
    # ============================================
    sl_distance = atr * ATR_SL_MULTIPLIER  # Default 1.5x ATR
    
    if direction == "long":
        stop_loss = entry_price - sl_distance
    else:
        stop_loss = entry_price + sl_distance
    
    sl_pct = (sl_distance / entry_price) * 100
    
    # Sanity check: SL must be between 0.3% and 3% for scalps
    if sl_pct < 0.3:
        sl_distance = entry_price * 0.003  # Floor at 0.3%
    elif sl_pct > 3.0:
        return None  # Reject — too volatile for a scalp
    
    # ============================================
    # TAKE PROFIT — Based on R multiples
    # ============================================
    tp1_distance = sl_distance * 1.5   # 1.5R — close 40% of position
    tp2_distance = sl_distance * 2.5   # 2.5R — close 40% of position
    tp3_distance = sl_distance * 4.0   # 4.0R — close remaining 20% (runner)
    
    if direction == "long":
        tp1 = entry_price + tp1_distance
        tp2 = entry_price + tp2_distance
        tp3 = entry_price + tp3_distance
    else:
        tp1 = entry_price - tp1_distance
        tp2 = entry_price - tp2_distance
        tp3 = entry_price - tp3_distance
    
    rr_ratio = tp2_distance / sl_distance  # Primary target R:R
    
    # Reject if R:R below minimum
    if rr_ratio < MIN_RR_RATIO:  # Default 2.0
        return None  # Signal rejected — risk/reward not worth it
    
    # ============================================
    # POSITION SIZING — % Risk Model
    # ============================================
    portfolio_value = get_portfolio_value()  # From exchange API or config
    risk_amount = portfolio_value * (RISK_PER_TRADE_PCT / 100)  # e.g., 1.5%
    
    # Adjust risk based on confluence score
    if score >= 85:
        risk_multiplier = 1.0      # Full risk for high confidence
    elif score >= 75:
        risk_multiplier = 0.75     # 75% risk for medium-high
    elif score >= 65:
        risk_multiplier = 0.5      # 50% risk for medium
    
    adjusted_risk = risk_amount * risk_multiplier
    position_size_usd = adjusted_risk / (sl_pct / 100)
    position_size_pct = (position_size_usd / portfolio_value) * 100
    
    # ============================================
    # PORTFOLIO HEAT CHECK
    # ============================================
    current_exposure = get_total_open_exposure()
    if current_exposure + position_size_pct > MAX_PORTFOLIO_HEAT_PCT:
        # Reduce position to fit within heat limit
        available_heat = MAX_PORTFOLIO_HEAT_PCT - current_exposure
        if available_heat < 1.0:
            return None  # No room — reject signal
        position_size_pct = available_heat
        position_size_usd = portfolio_value * (position_size_pct / 100)
    
    # ============================================
    # TRAILING STOP LOGIC (after TP1 hit)
    # ============================================
    trailing_stop_rules = {
        "after_tp1": "Move SL to breakeven (entry price)",
        "after_tp2": "Trail SL to TP1 level",
        "trail_method": "ATR trailing: SL follows price by 1x ATR distance",
        "max_hold_candles": SCALP_MAX_HOLD_TIME_CANDLES  # 8 candles = 2 hours
    }
    
    return {
        "stop_loss": round(stop_loss, get_price_precision(pair)),
        "tp1": round(tp1, get_price_precision(pair)),
        "tp2": round(tp2, get_price_precision(pair)),
        "tp3": round(tp3, get_price_precision(pair)),
        "sl_pct": round(sl_pct, 2),
        "rr_ratio": round(rr_ratio, 1),
        "position_size_usd": round(position_size_usd, 2),
        "position_size_pct": round(position_size_pct, 2),
        "risk_amount_usd": round(adjusted_risk, 2),
        "trailing_stop": trailing_stop_rules,
        "exit_plan": {
            "tp1_close_pct": 40,   # Close 40% at TP1
            "tp2_close_pct": 40,   # Close 40% at TP2
            "tp3_close_pct": 20    # Close 20% at TP3 (runner)
        }
    }
```

---

## 9. ANTI-EMOTION SAFEGUARDS & CIRCUIT BREAKERS

### 9.1 Objective

Prevent the system (and the trader following signals) from making emotional decisions. These are hard-coded rules that CANNOT be overridden during live trading.

### 9.2 Build Logic

**File:** `risk/circuit_breakers.py`

```python
class CircuitBreakers:
    """
    Every method returns: (should_block: bool, reason: str)
    If ANY breaker triggers, the signal is BLOCKED.
    """
    
    def check_all(self, pair, direction):
        """Run all circuit breakers. Block if ANY triggers."""
        checks = [
            self.check_revenge_trade(pair),
            self.check_overtrading(),
            self.check_daily_drawdown(),
            self.check_win_streak(),
            self.check_correlation(pair, direction),
            self.check_time_filter(),
            self.check_max_positions(),
            self.check_volatility_extreme(pair),
            self.check_macro_danger()
        ]
        
        for blocked, reason in checks:
            if blocked:
                logger.warning(f"CIRCUIT BREAKER: {reason}")
                self.send_breaker_alert(reason)
                return True, reason
        
        return False, None
    
    # =============================================
    # BREAKER 1: Revenge Trade Prevention
    # =============================================
    def check_revenge_trade(self, pair):
        """
        After a stop-loss hit on a pair, block new signals on the SAME pair
        for 15 minutes (1 candle). Prevents emotional re-entry.
        """
        last_sl_hit = db.get_last_sl_hit(pair)
        if last_sl_hit and (now() - last_sl_hit) < timedelta(seconds=REVENGE_TRADE_COOLDOWN_SEC):
            remaining = REVENGE_TRADE_COOLDOWN_SEC - (now() - last_sl_hit).seconds
            return True, f"Revenge trade blocked on {pair}. Cooldown: {remaining}s remaining"
        return False, None
    
    # =============================================
    # BREAKER 2: Overtrading Limiter
    # =============================================
    def check_overtrading(self):
        """
        Max 10 signals per hour. Prevents signal spam during volatile periods.
        Quality > quantity.
        """
        signals_last_hour = db.count_signals_since(now() - timedelta(hours=1))
        if signals_last_hour >= MAX_SIGNALS_PER_HOUR:
            return True, f"Overtrading limit reached: {signals_last_hour}/{MAX_SIGNALS_PER_HOUR} signals this hour"
        return False, None
    
    # =============================================
    # BREAKER 3: Daily Drawdown Circuit Breaker
    # =============================================
    def check_daily_drawdown(self):
        """
        If total realized + unrealized P&L today exceeds -5%, PAUSE ALL SIGNALS.
        Send alert to admin. Require manual /resume command to restart.
        
        This is the most important breaker. It prevents catastrophic loss days.
        """
        daily_pnl_pct = db.get_daily_pnl_percentage()
        if daily_pnl_pct <= -DRAWDOWN_CIRCUIT_BREAKER_PCT:
            self.pause_all_signals()
            return True, f"DAILY DRAWDOWN BREAKER: {daily_pnl_pct:.1f}% loss today. All signals PAUSED."
        return False, None
    
    # =============================================
    # BREAKER 4: Win Streak Cooldown
    # =============================================
    def check_win_streak(self):
        """
        After 5+ consecutive wins, reduce position size by 50%.
        Overconfidence leads to oversized bets and eventual blowup.
        
        Not a full block — just reduces size.
        """
        streak = db.get_current_win_streak()
        if streak >= WIN_STREAK_COOLDOWN_THRESHOLD:
            # Don't block, but flag for position size reduction
            self.position_size_modifier = 0.5
            return False, None  # Allow but with reduced size
        self.position_size_modifier = 1.0
        return False, None
    
    # =============================================
    # BREAKER 5: Correlation Guard
    # =============================================
    def check_correlation(self, pair, direction):
        """
        Prevent overexposure to correlated assets.
        e.g., Don't go long on ETH, MATIC, ARB, and OP simultaneously —
        they all move together. One ETH dump kills all 4 positions.
        
        Implementation:
        1. Calculate 30-day rolling correlation between all open position pairs
        2. If new signal pair has >0.85 correlation with existing position = BLOCK
        3. Max 2 positions in same correlation cluster
        """
        open_positions = db.get_open_positions()
        for pos in open_positions:
            if pos.direction == direction:  # Same direction increases risk
                corr = calculate_correlation(pair, pos.pair, period=30, timeframe="1h")
                if corr > CORRELATION_BLOCK_THRESHOLD:
                    return True, f"Correlation block: {pair} correlates {corr:.2f} with open {pos.pair}"
        return False, None
    
    # =============================================
    # BREAKER 6: Time of Day Filter
    # =============================================
    def check_time_filter(self):
        """
        Crypto trades 24/7 but certain hours have better signal quality.
        
        Optimal scalping windows (UTC):
        - 08:00-12:00 (London session open, high volume)
        - 13:00-17:00 (US session, highest volume)
        - 00:00-02:00 (Asia session open, good for Asian-heavy tokens)
        
        Low quality hours (UTC):
        - 04:00-07:00 (dead zone between Asia close and London open)
        - 20:00-23:00 (US winding down, low volume)
        
        During low-quality hours: increase confidence threshold to 85 (from 80)
        """
        hour_utc = datetime.utcnow().hour
        low_quality_hours = list(range(4, 8)) + list(range(20, 24))
        if hour_utc in low_quality_hours:
            self.confidence_threshold_modifier = 5  # Raise threshold by 5 points
        else:
            self.confidence_threshold_modifier = 0
        return False, None  # Never fully blocks, just raises bar
    
    # =============================================
    # BREAKER 7: Max Concurrent Positions
    # =============================================
    def check_max_positions(self):
        """Hard limit on open positions. Default: 5."""
        open_count = db.count_open_positions()
        if open_count >= MAX_CONCURRENT_POSITIONS:
            return True, f"Max positions reached: {open_count}/{MAX_CONCURRENT_POSITIONS}"
        return False, None
    
    # =============================================
    # BREAKER 8: Extreme Volatility Guard
    # =============================================
    def check_volatility_extreme(self, pair):
        """
        If ATR is >3x the 20-period ATR average, the pair is in chaos mode.
        Scalping in extreme volatility = gambling. Skip it.
        """
        current_atr = get_current_atr(pair)
        avg_atr = get_avg_atr(pair, periods=20)
        if current_atr > avg_atr * 3:
            return True, f"Extreme volatility on {pair}: ATR {current_atr:.4f} vs avg {avg_atr:.4f}"
        return False, None
    
    # =============================================
    # BREAKER 9: Macro Danger Alert
    # =============================================
    def check_macro_danger(self):
        """
        If a major macro event is causing market-wide panic:
        - S&P futures drop >2% in 1 hour
        - DXY spike >1% in 1 hour
        - BTC drops >5% in 1 hour
        
        PAUSE all signals. Send admin alert.
        """
        btc_1h_change = get_price_change("BTC/USDT", "1h")
        if abs(btc_1h_change) > 5.0:
            return True, f"MACRO DANGER: BTC moved {btc_1h_change:.1f}% in 1 hour. All signals paused."
        return False, None
```

---

## 10. TELEGRAM BOT — SIGNAL DELIVERY

### 10.1 Objective

Deliver formatted, data-rich signals to Telegram. The trader should see EXACTLY why the system is suggesting this trade, with full data transparency.

### 10.2 Build Logic

**File:** `telegram_bot/bot.py`

```python
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Register command handlers
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.reply("Scalp Engine Active. Use /help for commands.")

@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    stats = db.get_performance_stats()
    await message.reply(format_stats(stats))

@dp.message(Command("pnl"))
async def cmd_pnl(message: types.Message):
    pnl = db.get_daily_pnl()
    await message.reply(format_pnl(pnl))

@dp.message(Command("winrate"))
async def cmd_winrate(message: types.Message):
    wr = db.get_winrate(period="7d")
    await message.reply(format_winrate(wr))

@dp.message(Command("drawdown"))
async def cmd_drawdown(message: types.Message):
    dd = db.get_max_drawdown()
    await message.reply(format_drawdown(dd))

@dp.message(Command("pause"))
async def cmd_pause(message: types.Message):
    if str(message.from_user.id) == TELEGRAM_ADMIN_ID:
        engine.pause()
        await message.reply("Signal generation PAUSED.")

@dp.message(Command("resume"))
async def cmd_resume(message: types.Message):
    if str(message.from_user.id) == TELEGRAM_ADMIN_ID:
        engine.resume()
        await message.reply("Signal generation RESUMED.")

@dp.message(Command("open"))
async def cmd_open(message: types.Message):
    positions = db.get_open_positions()
    await message.reply(format_open_positions(positions))

@dp.message(Command("heat"))
async def cmd_heat(message: types.Message):
    heat = risk_manager.get_portfolio_heat()
    await message.reply(f"Portfolio Heat: {heat:.1f}% / {MAX_PORTFOLIO_HEAT_PCT}%")
```

### 10.3 Signal Message Format

**File:** `telegram_bot/formatter.py`

```python
def format_signal(signal: dict) -> str:
    """
    Format the signal into a Telegram message.
    Every piece of data must be visible. Full transparency.
    """
    
    direction_emoji = "🟢" if signal["direction"] == "long" else "🔴"
    direction_text = "LONG" if signal["direction"] == "long" else "SHORT"
    tier_emoji = "🔥" if signal["signal_tier"] == "HIGH" else "⚡"
    
    # Confidence stars
    score = signal["confluence_score"]
    if score >= 90: stars = "⭐⭐⭐⭐⭐"
    elif score >= 80: stars = "⭐⭐⭐⭐"
    elif score >= 70: stars = "⭐⭐⭐"
    else: stars = "⭐⭐"
    
    # Build data breakdown section
    bd = signal["data_breakdown"]
    
    msg = f"""
{direction_emoji} {direction_text} — {signal['pair']} {tier_emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Confluence Score: {score}/100 {stars}
⏱ Timeframe: 15m | Valid: {SIGNAL_EXPIRY_MINUTES} min

💰 TRADE SETUP
├─ Entry: ${signal['entry_price']}
├─ Stop Loss: ${signal['stop_loss']} ({signal['sl_pct']}%)
├─ TP1: ${signal['tp1']} → Close 40% | Move SL to BE
├─ TP2: ${signal['tp2']} → Close 40% | Trail SL to TP1
└─ TP3: ${signal['tp3']} → Close 20% (runner)

⚖️ Risk/Reward: 1:{signal['rr_ratio']}
📐 Position Size: {signal['position_size_pct']}% of portfolio
💵 Risk: ${signal['risk_amount_usd']}

📡 DATA BREAKDOWN
├─ TA: {bd['technical']['score']}/100
│   └─ {', '.join(bd['technical']['reasons'][:2])}
├─ Order Flow: {bd['orderflow']['score']}/100
│   └─ {', '.join(bd['orderflow']['reasons'][:2])}
├─ On-Chain: {bd['onchain']['score']}/100
│   └─ {', '.join(bd['onchain']['reasons'][:1])}
├─ Sentiment: {bd['sentiment']['score']}/100
│   └─ {', '.join(bd['sentiment']['reasons'][:1])}
└─ Backtest: {bd['backtest']['score']}/100
    └─ Historical win rate: {bd['backtest']['win_rate']}%

🌐 Market Regime: {signal['market_regime']}
━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Max hold: {SCALP_MAX_HOLD_TIME_CANDLES} candles (2h)
🤖 100% data-driven. Zero emotion.
"""
    return msg.strip()
```

### 10.4 Additional Alert Types

```python
# WHALE ALERT
def format_whale_alert(event):
    return f"""
🐋 WHALE ALERT — {event['token']}
━━━━━━━━━━━━━━━━━━━
{event['amount_usd']} {event['direction']}
From: {event['from_label']}
To: {event['to_label']}
Potential Impact: {event['impact']}
"""

# CIRCUIT BREAKER ALERT
def format_breaker_alert(reason):
    return f"""
🚨 CIRCUIT BREAKER ACTIVATED
━━━━━━━━━━━━━━━━━━━
Reason: {reason}
Action: Signals PAUSED
Resume: Use /resume command
"""

# DAILY PERFORMANCE REPORT (auto-sent at 00:00 UTC)
def format_daily_report(stats):
    return f"""
📊 DAILY REPORT — {stats['date']}
━━━━━━━━━━━━━━━━━━━
Signals Sent: {stats['total_signals']}
Wins: {stats['wins']} | Losses: {stats['losses']}
Win Rate: {stats['win_rate']}%
Total P&L: {stats['pnl_pct']}%
Best Trade: {stats['best_trade']}
Worst Trade: {stats['worst_trade']}
Max Drawdown: {stats['max_drawdown']}%
Avg R:R Achieved: {stats['avg_rr']}
"""
```

---

## 11. DATABASE & LOGGING SYSTEM

### 11.1 Schema

**File:** `database/schema.sql`

```sql
-- Signals table — every signal generated (sent or discarded)
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL,
    confluence_score INTEGER NOT NULL,
    signal_tier VARCHAR(10),
    entry_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    tp1 DECIMAL(20, 8),
    tp2 DECIMAL(20, 8),
    tp3 DECIMAL(20, 8),
    rr_ratio DECIMAL(5, 2),
    position_size_pct DECIMAL(5, 2),
    
    -- Data layer scores
    ta_score INTEGER,
    orderflow_score INTEGER,
    onchain_score INTEGER,
    sentiment_score INTEGER,
    backtest_score INTEGER,
    
    -- Outcome tracking
    outcome VARCHAR(10),
    actual_pnl_pct DECIMAL(10, 4),
    actual_rr DECIMAL(5, 2),
    hold_duration_minutes INTEGER,
    
    -- Metadata
    market_regime VARCHAR(20),
    block_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    valid_until TIMESTAMP
);

-- Open positions tracker
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(id),
    pair VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL,
    entry_price DECIMAL(20, 8),
    current_price DECIMAL(20, 8),
    unrealized_pnl_pct DECIMAL(10, 4),
    stop_loss DECIMAL(20, 8),
    status VARCHAR(10) DEFAULT 'open',
    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP
);

-- Performance tracking
CREATE TABLE daily_performance (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_signals INTEGER DEFAULT 0,
    signals_sent INTEGER DEFAULT 0,
    signals_blocked INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    total_pnl_pct DECIMAL(10, 4),
    max_drawdown_pct DECIMAL(10, 4),
    avg_rr_achieved DECIMAL(5, 2),
    best_trade_pnl DECIMAL(10, 4),
    worst_trade_pnl DECIMAL(10, 4),
    breaker_activations INTEGER DEFAULT 0
);

-- Whale events log
CREATE TABLE whale_events (
    id SERIAL PRIMARY KEY,
    token VARCHAR(20),
    amount_usd DECIMAL(20, 2),
    direction VARCHAR(20),
    from_label VARCHAR(100),
    to_label VARCHAR(100),
    blockchain VARCHAR(20),
    tx_hash VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Sentiment snapshots
CREATE TABLE sentiment_snapshots (
    id SERIAL PRIMARY KEY,
    fear_greed_index INTEGER,
    market_regime VARCHAR(20),
    btc_dominance DECIMAL(5, 2),
    social_volume_score INTEGER,
    macro_status VARCHAR(20),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Circuit breaker log
CREATE TABLE breaker_events (
    id SERIAL PRIMARY KEY,
    breaker_type VARCHAR(50),
    reason TEXT,
    pair VARCHAR(20),
    action_taken VARCHAR(20),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_signals_pair_created ON signals(pair, created_at);
CREATE INDEX idx_signals_outcome ON signals(outcome);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_daily_perf_date ON daily_performance(date);
```

### 11.2 Trade Outcome Tracking

```python
async def track_signal_outcomes():
    """
    Background task that runs every 60 seconds.
    Checks if any open signal hit TP or SL.
    """
    open_signals = db.get_unresolved_signals()
    
    for signal in open_signals:
        current_price = get_current_price(signal.pair)
        
        if signal.direction == "long":
            if current_price <= signal.stop_loss:
                close_signal(signal, outcome="sl", exit_price=signal.stop_loss)
            elif current_price >= signal.tp3:
                close_signal(signal, outcome="tp3", exit_price=signal.tp3)
            elif current_price >= signal.tp2:
                close_signal(signal, outcome="tp2", exit_price=signal.tp2)
            elif current_price >= signal.tp1:
                update_trailing_stop(signal, new_sl=signal.entry_price)
        
        # Check expiry
        if datetime.utcnow() > signal.valid_until:
            close_signal(signal, outcome="expired", exit_price=current_price)
        
        # Check max hold time
        candles_held = (datetime.utcnow() - signal.created_at).seconds / 900
        if candles_held >= SCALP_MAX_HOLD_TIME_CANDLES:
            close_signal(signal, outcome="max_hold", exit_price=current_price)
```

---

## 12. BACKTESTING MODULE

### 12.1 Objective

Before going live, every strategy configuration must be backtested on at least 3 months of historical data.

### 12.2 Build Logic

**File:** `backtesting/replay_engine.py`

```python
class BacktestEngine:
    """
    Replays historical data through the signal engine to validate profitability.
    
    Requirements:
    1. Must use SAME code as live engine (no separate backtest logic)
    2. Must account for slippage (0.05% per trade)
    3. Must account for fees (0.1% maker/taker)
    4. Must respect all circuit breakers
    5. Walk-forward optimization (train on 2 months, test on 1 month)
    """
    
    def run_backtest(self, pairs, start_date, end_date, initial_capital=10000):
        # Download data
        for pair in pairs:
            data[pair] = exchange.fetch_ohlcv(pair, "15m", since=start_date)
        
        # Replay
        portfolio = initial_capital
        trades = []
        
        for timestamp in all_timestamps:
            for pair in pairs:
                candle = data[pair][timestamp]
                signal = confluence_engine.process(pair, candle)
                
                if signal and signal["action"] != "DISCARD":
                    entry = signal["entry_price"] * (1 + 0.0005)  # 0.05% slippage
                    fee = entry * 0.001  # 0.1% fee
                    
                    outcome = simulate_trade_outcome(pair, entry, signal["stop_loss"],
                                                      signal["tp1"], signal["tp2"], signal["tp3"],
                                                      future_candles)
                    trades.append(outcome)
        
        return self.calculate_metrics(trades, initial_capital)
    
    def calculate_metrics(self, trades, initial_capital):
        return {
            "total_trades": len(trades),
            "win_rate": wins / len(trades) * 100,
            "avg_rr_achieved": mean([t.rr for t in trades]),
            "total_pnl_pct": (final_portfolio - initial_capital) / initial_capital * 100,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "profit_factor": gross_profit / gross_loss,
            "avg_hold_time_minutes": mean([t.hold_time for t in trades]),
            "best_trade": max(trades, key=lambda t: t.pnl),
            "worst_trade": min(trades, key=lambda t: t.pnl),
            "consecutive_losses_max": max_consecutive_losses,
            "monthly_returns": monthly_breakdown
        }
```

### 12.3 Minimum Performance Requirements Before Going Live

| Metric | Minimum Threshold |
|---|---|
| Win Rate | > 55% |
| Profit Factor | > 1.5 |
| Max Drawdown | < 15% |
| Sharpe Ratio | > 1.2 |
| Avg R:R Achieved | > 1.5 |
| Max Consecutive Losses | < 8 |
| Sample Size | > 200 trades |

---

## 13. DEPLOYMENT & INFRASTRUCTURE

### 13.1 Minimum Server Requirements

| Resource | Requirement |
|---|---|
| CPU | 4 cores |
| RAM | 8 GB minimum (16 GB recommended) |
| Storage | 50 GB SSD |
| Network | Low latency to Binance servers |
| OS | Ubuntu 22.04 LTS |

### 13.2 Docker Compose

```yaml
version: '3.8'

services:
  scalp-engine:
    build: .
    restart: always
    env_file: .env
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    networks:
      - scalp-net

  postgres:
    image: postgres:16
    restart: always
    environment:
      POSTGRES_DB: scalp_engine
      POSTGRES_USER: scalp
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    networks:
      - scalp-net

  redis:
    image: redis:7-alpine
    restart: always
    networks:
      - scalp-net

volumes:
  pgdata:

networks:
  scalp-net:
```

### 13.3 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### 13.4 Health Monitoring

```python
async def health_check():
    """Runs every 5 minutes"""
    checks = {
        "binance_ws": check_websocket_connected(),
        "database": check_db_connection(),
        "redis": check_redis_connection(),
        "last_signal_age": get_time_since_last_signal(),
        "memory_usage": get_memory_percent(),
        "cpu_usage": get_cpu_percent()
    }
    
    if any_check_failed(checks):
        send_admin_alert(f"HEALTH CHECK FAILED: {checks}")
    
    if not checks["binance_ws"]:
        await restart_websocket_connections()
```

---

## 14. API KEYS & EXTERNAL SERVICES CHECKLIST

| Service | Purpose | Cost | URL |
|---|---|---|---|
| Binance API | Price data, order book, trades | Free | binance.com/en/my/settings/api-management |
| Telegram Bot | Signal delivery | Free | t.me/BotFather |
| Whale Alert | Whale transfers | Free tier: 10 req/min | whale-alert.io |
| Etherscan | ERC-20 token tracking | Free tier: 5 req/sec | etherscan.io/apis |
| LunarCrush | Social sentiment data | Free tier available | lunarcrush.com/developers |
| Alternative.me | Fear & Greed Index | Free | alternative.me/crypto/api/ |
| CryptoQuant (Optional) | Exchange flows, on-chain | Paid ($50+/mo) | cryptoquant.com |
| Glassnode (Optional) | Advanced on-chain metrics | Paid ($30+/mo) | glassnode.com |
| Alpha Vantage (Optional) | S&P 500, DXY data | Free tier: 25 req/day | alphavantage.co |

---

## 15. DEVELOPMENT ROADMAP

### Phase 1: Foundation (Week 1-2)

- Project scaffolding and configuration
- Binance WebSocket connection (all USDT pairs)
- Market scanner with filtering
- Full TA engine (all indicators + patterns)
- Basic signal generation (TA only)
- Risk management (SL/TP/position sizing)
- Telegram bot with signal delivery
- Database setup and signal logging

**Milestone:** Bot sends TA-only signals to Telegram with correct formatting.

### Phase 2: Order Flow + On-Chain (Week 3-4)

- Order book depth analysis
- Trade flow aggressor tracking
- Funding rate & OI integration
- Whale movement tracking
- Exchange flow monitoring
- Integrate into confluence scoring

**Milestone:** Signals now include Order Flow and On-Chain scores.

### Phase 3: Sentiment + Confluence (Week 5)

- Fear & Greed integration
- Social sentiment scoring
- Market regime classification
- Macro correlation tracker
- Full confluence scoring engine
- All circuit breakers implemented

**Milestone:** Full 4-layer confluence scoring live. All circuit breakers active.

### Phase 4: Backtesting + Optimization (Week 6)

- Historical data downloader
- Backtest replay engine
- Performance metrics calculator
- Walk-forward optimization
- Parameter tuning based on backtest results
- Validate minimum performance thresholds

**Milestone:** Backtest confirms >55% win rate, >1.5 profit factor.

### Phase 5: Live Monitoring (Week 7+)

- Paper trading mode (signals without real money)
- Performance tracking dashboard
- Daily report automation
- Strategy drift detection
- Gradual ramp-up to live capital

---

## CRITICAL REMINDERS FOR THE PROGRAMMER

1. **Every number must come from data.** No hardcoded entries based on "feeling." If you can't quantify it, don't use it.

2. **The 15m candle close is the trigger.** All analysis runs on candle close, never on tick data. This prevents whipsaw signals from wicks.

3. **Fail safe, not fail open.** If any data source is unavailable, REDUCE confidence, don't ignore the missing data. A score calculated from 3/4 layers should be penalized.

4. **Log everything.** Every signal generated, every signal discarded, every circuit breaker activation, every API failure. Debugging requires complete audit trail.

5. **Never trust a single indicator.** The entire point of confluence is that NO single data point triggers a trade. Minimum 3 indicators must agree.

6. **Scalping = speed.** The entire signal generation pipeline (scanner → TA → order flow → confluence → risk → telegram) must complete within 5 seconds of candle close.

7. **Paper trade for minimum 2 weeks** before using real capital. Track every signal as if real money was on the line.

8. **The system is the boss.** If the circuit breaker says stop, you stop. No overrides. No "just this once." The system exists to protect against human weakness.

---

*Built for the 15-minute timeframe. Designed to eliminate emotion. Powered by data confluence. Every trade is a calculated probability, not a gamble.*
