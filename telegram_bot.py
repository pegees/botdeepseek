#!/usr/bin/env python3
"""
TELEGRAM SCALP SCANNER BOT
==========================
Chart patterns + Leverage gains + Chart images

Commands:
  /scan       - Full scan (top 30 pairs)
  /scan BTC   - Deep scan single pair
  /scan meme  - Volatile pairs only
  /top        - Quick top 10 movers
  /status     - Bot health check
  /autoscan   - Auto-scan every 15 min
  /leverage   - Set leverage for gain calculation
  /ask        - Chat with DeepSeek AI
  /clear      - Clear chat history
  /prompt     - Customize scan instructions
  /resetprompt - Reset to default prompt
"""
import os
import re
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Fix matplotlib backend BEFORE importing mplfinance
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Load API keys - from .env file locally, from environment on Railway
from dotenv import load_dotenv
env_file = Path(__file__).parent / "API_KEYS.env"
if env_file.exists():
    load_dotenv(env_file)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Validate critical keys
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set! Add it to API_KEYS.env or Railway variables.")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not set!")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports for analysis
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import mplfinance as mpf

# Config
TOP_PAIRS_TO_SCAN = 30
TIMEFRAME = "15m"
CANDLES_TO_FETCH = 200
DEFAULT_LEVERAGE = 50

# Bot setup
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()
executor = ThreadPoolExecutor(max_workers=2)


@dp.update.outer_middleware()
async def log_all_updates(handler, event, data):
    """Log ALL incoming updates for debugging"""
    logger.info(f"RAW UPDATE: type={event.event_type} data={event.model_dump_json()[:300]}")
    return await handler(event, data)


# State
autoscan_enabled = False
last_scan_time = None
is_scanning = False
user_leverage = DEFAULT_LEVERAGE
df_cache = {}  # Cache dataframes for chart generation

# DeepSeek chat state
deepseek_chat_history = []  # Conversation memory for /ask
custom_trading_prompt = ""  # User's custom instructions for scan prompt


def get_exchange():
    """Initialize Binance connection"""
    config = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {
            "defaultType": "spot",
            "fetchCurrencies": False,
        }
    }
    return ccxt.binance(config)


exchange = get_exchange()


# ============================================
# CHART PATTERN DETECTION
# ============================================

def calculate_slope(values):
    """Linear regression slope"""
    if len(values) < 2:
        return 0
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    return slope


def detect_chart_pattern(df, lookback=50):
    """Detect chart patterns from swing highs/lows"""
    recent = df.tail(lookback).copy()
    close = recent["close"].values
    high = recent["high"].values
    low = recent["low"].values

    # Find swing points
    swing_high_idx = argrelextrema(high, np.greater, order=5)[0]
    swing_low_idx = argrelextrema(low, np.less, order=5)[0]

    swing_highs = [(i, high[i]) for i in swing_high_idx]
    swing_lows = [(i, low[i]) for i in swing_low_idx]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"pattern": "No Clear Pattern", "confidence": 0, "direction_bias": "neutral", "description": ""}

    high_slope = calculate_slope([h[1] for h in swing_highs])
    low_slope = calculate_slope([l[1] for l in swing_lows])

    price_range = max(high) - min(low)
    if price_range == 0:
        return {"pattern": "No Clear Pattern", "confidence": 0, "direction_bias": "neutral", "description": ""}

    hs = high_slope / price_range
    ls = low_slope / price_range
    current_price = close[-1]

    # Ranging Channel
    if abs(hs) < 0.05 and abs(ls) < 0.05:
        range_pos = (current_price - min(low)) / price_range
        if range_pos > 0.8:
            bias = "short"
        elif range_pos < 0.2:
            bias = "long"
        else:
            bias = "neutral"
        return {"pattern": "Ranging Channel", "confidence": 75, "direction_bias": bias,
                "description": f"Price ranging between {min(low):.6f} and {max(high):.6f}"}

    # Ascending Channel
    if hs > 0.03 and ls > 0.03 and abs(hs - ls) < 0.05:
        return {"pattern": "Ascending Channel", "confidence": 70, "direction_bias": "long",
                "description": "Parallel uptrend channel"}

    # Descending Channel
    if hs < -0.03 and ls < -0.03 and abs(hs - ls) < 0.05:
        return {"pattern": "Descending Channel", "confidence": 70, "direction_bias": "short",
                "description": "Parallel downtrend channel"}

    # Ascending Triangle
    if abs(hs) < 0.03 and ls > 0.05:
        return {"pattern": "Ascending Triangle", "confidence": 80, "direction_bias": "long",
                "description": "Flat resistance + rising support — bullish breakout likely"}

    # Descending Triangle
    if abs(ls) < 0.03 and hs < -0.05:
        return {"pattern": "Descending Triangle", "confidence": 80, "direction_bias": "short",
                "description": "Flat support + falling resistance — bearish breakdown likely"}

    # Symmetrical Triangle
    if hs < -0.03 and ls > 0.03:
        return {"pattern": "Symmetrical Triangle", "confidence": 65, "direction_bias": "neutral",
                "description": "Converging trendlines — breakout imminent"}

    # Rising Wedge
    if hs > 0.02 and ls > 0.02 and ls > hs:
        return {"pattern": "Rising Wedge", "confidence": 70, "direction_bias": "short",
                "description": "Converging uptrend — bearish reversal likely"}

    # Falling Wedge
    if hs < -0.02 and ls < -0.02 and ls > hs:
        return {"pattern": "Falling Wedge", "confidence": 70, "direction_bias": "long",
                "description": "Converging downtrend — bullish reversal likely"}

    # Double Top
    if len(swing_highs) >= 2:
        h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
        if abs(h1 - h2) / h1 < 0.005 and current_price < h2 * 0.995:
            return {"pattern": "Double Top", "confidence": 75, "direction_bias": "short",
                    "description": f"Two rejections at ~{h1:.6f} — bearish reversal"}

    # Double Bottom
    if len(swing_lows) >= 2:
        l1, l2 = swing_lows[-2][1], swing_lows[-1][1]
        if abs(l1 - l2) / l1 < 0.005 and current_price > l2 * 1.005:
            return {"pattern": "Double Bottom", "confidence": 75, "direction_bias": "long",
                    "description": f"Two bounces at ~{l1:.6f} — bullish reversal"}

    # Bull Flag
    mid = len(close) // 2
    pre_move = (close[mid] - close[0]) / close[0] if close[0] > 0 else 0
    flag_move = (close[-1] - close[mid]) / close[mid] if close[mid] > 0 else 0
    if pre_move > 0.02 and -0.015 < flag_move < 0:
        return {"pattern": "Bull Flag", "confidence": 70, "direction_bias": "long",
                "description": "Strong rally + pullback — continuation likely"}

    # Bear Flag
    if pre_move < -0.02 and 0 < flag_move < 0.015:
        return {"pattern": "Bear Flag", "confidence": 70, "direction_bias": "short",
                "description": "Strong drop + bounce — continuation down likely"}

    return {"pattern": "No Clear Pattern", "confidence": 0, "direction_bias": "neutral", "description": ""}


# ============================================
# LEVERAGE CALCULATION
# ============================================

def calculate_leveraged_gains(entry, sl, tp1, tp2, tp3, direction, leverage):
    """Calculate leveraged % gains"""
    if direction == "long":
        raw_sl = (sl - entry) / entry * 100
        raw_tp1 = (tp1 - entry) / entry * 100
        raw_tp2 = (tp2 - entry) / entry * 100
        raw_tp3 = (tp3 - entry) / entry * 100
        liquidation = entry * (1 - (1 / leverage) * 0.95)
    else:
        raw_sl = (entry - sl) / entry * 100
        raw_tp1 = (entry - tp1) / entry * 100
        raw_tp2 = (entry - tp2) / entry * 100
        raw_tp3 = (entry - tp3) / entry * 100
        liquidation = entry * (1 + (1 / leverage) * 0.95)

    raw_sl = -abs(raw_sl)

    return {
        "leverage": leverage,
        "sl_raw": round(raw_sl, 2),
        "sl_leveraged": round(raw_sl * leverage, 1),
        "tp1_raw": round(raw_tp1, 2),
        "tp1_leveraged": round(raw_tp1 * leverage, 1),
        "tp2_raw": round(raw_tp2, 2),
        "tp2_leveraged": round(raw_tp2 * leverage, 1),
        "tp3_raw": round(raw_tp3, 2),
        "tp3_leveraged": round(raw_tp3 * leverage, 1),
        "liquidation_price": round(liquidation, 8)
    }


def add_leverage_to_result(deepseek_result, leverage):
    """Parse DeepSeek result and add leverage calculations"""
    entry_match = re.search(r'Entry[:\s]*\$?([\d,.]+)', deepseek_result, re.IGNORECASE)
    sl_match = re.search(r'Stop Loss[:\s]*\$?([\d,.]+)', deepseek_result, re.IGNORECASE)
    tp1_match = re.search(r'TP1[:\s]*\$?([\d,.]+)', deepseek_result, re.IGNORECASE)
    tp2_match = re.search(r'TP2[:\s]*\$?([\d,.]+)', deepseek_result, re.IGNORECASE)
    tp3_match = re.search(r'TP3[:\s]*\$?([\d,.]+)', deepseek_result, re.IGNORECASE)

    if not all([entry_match, sl_match, tp1_match]):
        return deepseek_result

    try:
        entry = float(entry_match.group(1).replace(",", ""))
        sl = float(sl_match.group(1).replace(",", ""))
        tp1 = float(tp1_match.group(1).replace(",", ""))
        tp2 = float(tp2_match.group(1).replace(",", "")) if tp2_match else tp1
        tp3 = float(tp3_match.group(1).replace(",", "")) if tp3_match else tp1
    except:
        return deepseek_result

    direction = "long" if tp1 > entry else "short"
    gains = calculate_leveraged_gains(entry, sl, tp1, tp2, tp3, direction, leverage)

    leverage_section = f"""
{'─' * 30}
LEVERAGE CALCULATION ({leverage}x)

At {leverage}x leverage:
- SL hit: {gains['sl_leveraged']}% loss (raw: {gains['sl_raw']}%)
- TP1 hit: +{gains['tp1_leveraged']}% gain (raw: +{gains['tp1_raw']}%)
- TP2 hit: +{gains['tp2_leveraged']}% gain (raw: +{gains['tp2_raw']}%)
- TP3 hit: +{gains['tp3_leveraged']}% gain (raw: +{gains['tp3_raw']}%)
- Liquidation: ${gains['liquidation_price']:.8f}

WARNING: {leverage}x = liquidation at ~{abs(round(1/leverage*100*0.95, 2))}% against you
"""
    return deepseek_result + leverage_section


# ============================================
# CHART IMAGE GENERATION
# ============================================

def generate_chart_image(df, symbol, pattern_name="", save_path="/tmp/chart.png"):
    """Generate candlestick chart with indicators"""
    try:
        chart_df = df.tail(80).copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], unit="ms")
        chart_df.set_index("timestamp", inplace=True)
        chart_df.index.name = "Date"

        add_plots = []

        # EMAs
        if "ema9" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["ema9"], color="yellow", width=1))
        if "ema21" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["ema21"], color="orange", width=1))
        if "ema50" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["ema50"], color="cyan", width=1))

        # Bollinger Bands
        if "bb_upper" in chart_df.columns and "bb_lower" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["bb_upper"], color="gray", width=0.7, linestyle="--"))
            add_plots.append(mpf.make_addplot(chart_df["bb_lower"], color="gray", width=0.7, linestyle="--"))

        # RSI
        if "rsi" in chart_df.columns:
            add_plots.append(mpf.make_addplot(chart_df["rsi"], panel=2, color="magenta", ylabel="RSI", ylim=(0, 100)))

        # Chart style
        mc = mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit", volume="in")
        style = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="nightclouds")

        title = f"{symbol} — 15m"
        if pattern_name:
            title += f" | {pattern_name}"

        mpf.plot(
            chart_df,
            type="candle",
            style=style,
            volume=True,
            addplot=add_plots if add_plots else None,
            title=title,
            figsize=(12, 8),
            savefig=save_path
        )
        return save_path
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return None


# ============================================
# DATA FETCHING & ANALYSIS
# ============================================

def get_top_volume_pairs(limit=30):
    """Get top pairs by 24h volume"""
    try:
        if not exchange.markets:
            exchange.load_markets()
    except Exception as e:
        logger.warning(f"load_markets failed: {e}, continuing anyway...")

    tickers = exchange.fetch_tickers()

    skip_bases = ["USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "EUR", "GBP"]
    pairs = []

    for symbol, t in tickers.items():
        if not symbol.endswith("/USDT"):
            continue
        if "/USDT:" in symbol:
            continue

        base = symbol.split("/")[0]
        if base in skip_bases:
            continue

        vol = t.get("quoteVolume") or 0
        if vol < 500_000:
            continue

        pairs.append({
            "symbol": symbol,
            "volume": vol,
            "price": t.get("last", 0),
            "change": t.get("percentage", 0) or 0
        })

    pairs.sort(key=lambda x: x["volume"], reverse=True)
    return pairs[:limit]


def analyze_pair(symbol):
    """Fetch candles, calculate indicators, detect pattern"""
    global df_cache

    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES_TO_FETCH)
    if len(ohlcv) < 100:
        return None

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Indicators
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["volume_sma"]

    # Cache for chart generation
    df_cache[symbol] = df.copy()

    c = df.iloc[-1]
    p = df.iloc[-2]

    # EMA structure
    ema_stack = "NONE"
    if pd.notna(c.get("ema9")) and pd.notna(c.get("ema21")) and pd.notna(c.get("ema50")):
        if c["ema9"] > c["ema21"] > c["ema50"]:
            ema_stack = "BULLISH (9>21>50)"
        elif c["ema9"] < c["ema21"] < c["ema50"]:
            ema_stack = "BEARISH (9<21<50)"
        else:
            ema_stack = "MIXED"

    # EMA cross
    ema_cross = "NONE"
    if pd.notna(c.get("ema9")) and pd.notna(p.get("ema9")):
        if p["ema9"] <= p["ema21"] and c["ema9"] > c["ema21"]:
            ema_cross = "BULLISH CROSS"
        elif p["ema9"] >= p["ema21"] and c["ema9"] < c["ema21"]:
            ema_cross = "BEARISH CROSS"

    # MACD
    macd_status = "N/A"
    hist = c.get("macd_hist")
    prev_hist = p.get("macd_hist")
    if pd.notna(hist) and pd.notna(prev_hist):
        if prev_hist < 0 and hist > 0:
            macd_status = "BULLISH CROSS"
        elif prev_hist > 0 and hist < 0:
            macd_status = "BEARISH CROSS"
        elif hist > 0:
            macd_status = "BULLISH"
        else:
            macd_status = "BEARISH"

    # Bollinger
    bb_pos = "N/A"
    bbu, bbl = c.get("bb_upper"), c.get("bb_lower")
    if pd.notna(bbu) and pd.notna(bbl) and bbu != bbl:
        pct = (c["close"] - bbl) / (bbu - bbl) * 100
        if pct < 20:
            bb_pos = f"LOWER BAND ({pct:.0f}%)"
        elif pct > 80:
            bb_pos = f"UPPER BAND ({pct:.0f}%)"
        else:
            bb_pos = f"MID ({pct:.0f}%)"

    # Volume
    vol_status = "NORMAL"
    vr = c.get("vol_ratio")
    if pd.notna(vr):
        if vr > 2:
            vol_status = f"SPIKE ({vr:.1f}x)"
        elif vr > 1.5:
            vol_status = f"HIGH ({vr:.1f}x)"
        elif vr < 0.5:
            vol_status = f"LOW ({vr:.1f}x)"

    atr_val = c.get("atr")
    atr_pct = (atr_val / c["close"] * 100) if pd.notna(atr_val) else 0

    # Chart pattern
    pattern = detect_chart_pattern(df)

    recent = df.tail(50)
    supports = sorted(recent["low"].nsmallest(3).unique())[:3]
    resistances = sorted(recent["high"].nlargest(3).unique(), reverse=True)[:3]

    return {
        "symbol": symbol,
        "price": round(c["close"], 8),
        "rsi": round(c["rsi"], 1) if pd.notna(c.get("rsi")) else "N/A",
        "ema_stack": ema_stack,
        "ema_cross": ema_cross,
        "macd_status": macd_status,
        "bollinger_position": bb_pos,
        "volume_status": vol_status,
        "volume_ratio": round(vr, 1) if pd.notna(vr) else "N/A",
        "atr": round(atr_val, 8) if pd.notna(atr_val) else "N/A",
        "atr_pct": round(atr_pct, 2),
        "supports": [round(s, 8) for s in supports],
        "resistances": [round(r, 8) for r in resistances],
        "chart_pattern": pattern["pattern"],
        "pattern_confidence": pattern["confidence"],
        "pattern_bias": pattern["direction_bias"],
        "pattern_description": pattern["description"]
    }


def scan_all_pairs(top_pairs):
    """Analyze all pairs"""
    results = []
    for i, pair_info in enumerate(top_pairs):
        symbol = pair_info["symbol"]
        try:
            analysis = analyze_pair(symbol)
            if analysis:
                analysis["volume_24h"] = pair_info["volume"]
                analysis["change_24h"] = pair_info["change"]
                results.append(analysis)
        except Exception as e:
            logger.error(f"Error {symbol}: {e}")
        time.sleep(0.3)
    return results


def build_deepseek_prompt(all_pair_data):
    """Build prompt with pattern info"""
    pair_summaries = ""
    for d in all_pair_data:
        pair_summaries += f"""
--- {d['symbol']} ---
Price: {d['price']} | 24h: {d.get('change_24h', 0):.1f}% | Vol: ${d.get('volume_24h', 0):,.0f}
RSI: {d['rsi']} | EMA: {d['ema_stack']} | Cross: {d['ema_cross']}
MACD: {d['macd_status']} | BB: {d['bollinger_position']}
Volume: {d['volume_status']} | ATR: {d['atr_pct']}%
Chart Pattern: {d['chart_pattern']} ({d['pattern_confidence']}% conf, bias: {d['pattern_bias']})
Pattern Detail: {d['pattern_description']}
Support: {d['supports'][:2]} | Resistance: {d['resistances'][:2]}
"""

    prompt = f"""You are an expert crypto scalp trader on 15-minute timeframe.

MARKET DATA ({len(all_pair_data)} pairs):
{pair_summaries}

Find TOP 1-3 BEST scalp setups. For each:
- PAIR + DIRECTION (LONG/SHORT)
- Chart Pattern identified
- CONFLUENCE (which 3+ indicators align)
- ENTRY price
- STOP LOSS (1.5x ATR)
- TP1 (1.5R - close 40%), TP2 (2.5R - close 40%), TP3 (4R - runner)
- CONFIDENCE (1-10)
- INVALIDATION

RULES:
- Need 3+ indicators aligned
- Skip LOW volume pairs
- Skip ATR < 0.3%
- Pay attention to chart patterns - they add confluence
- If nothing good: say "NO SETUP - WAIT"
{f'{chr(10)}ADDITIONAL USER INSTRUCTIONS:{chr(10)}{custom_trading_prompt}' if custom_trading_prompt else ''}

Be concise. Format clearly."""
    return prompt


def ask_deepseek(prompt):
    """Send to DeepSeek API"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a professional crypto scalp trader. Be concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        return None

    return response.json()["choices"][0]["message"]["content"]


def extract_pair_from_result(result):
    """Find which pair DeepSeek recommended"""
    match = re.search(r'(?:LONG|SHORT)\s*[-—]\s*(\w+/USDT)', result, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'(\w+/USDT)', result)
    return match.group(1) if match else None


def blocking_scan(mode="all", specific_pair=None):
    """Blocking scan function"""
    global last_scan_time, df_cache
    df_cache = {}

    if specific_pair:
        symbol = specific_pair.upper()
        if not symbol.endswith("/USDT"):
            symbol = f"{symbol}/USDT"

        analysis = analyze_pair(symbol)
        if not analysis:
            return f"Could not analyze {symbol}", None

        analysis["volume_24h"] = 0
        analysis["change_24h"] = 0
        prompt = build_deepseek_prompt([analysis])
        result = ask_deepseek(prompt)
        pair_count = 1
        recommended_pair = symbol
    else:
        top_pairs = get_top_volume_pairs(TOP_PAIRS_TO_SCAN)

        if mode == "meme":
            top_pairs = [p for p in top_pairs if abs(p.get("change", 0)) > 3]
            if not top_pairs:
                return "No volatile pairs (>3% change) found.", None

        if not top_pairs:
            return "No pairs found.", None

        all_data = scan_all_pairs(top_pairs)
        if not all_data:
            return "Could not analyze any pairs.", None

        prompt = build_deepseek_prompt(all_data)
        result = ask_deepseek(prompt)
        pair_count = len(all_data)
        recommended_pair = extract_pair_from_result(result) if result else None

    if not result:
        return "DeepSeek returned no response.", None

    # Add leverage
    result = add_leverage_to_result(result, user_leverage)

    last_scan_time = datetime.now(timezone.utc)
    timestamp = last_scan_time.strftime("%Y-%m-%d %H:%M UTC")

    header = f"SCALP SCAN - {timestamp}\n"
    header += f"Timeframe: 15m | Pairs: {pair_count} | Leverage: {user_leverage}x\n"
    if mode == "meme":
        header += "Mode: VOLATILE ONLY\n"
    header += "\n"

    # Generate chart
    chart_path = None
    if recommended_pair and recommended_pair in df_cache:
        pattern_name = ""
        for d in (all_data if not specific_pair else [analysis]):
            if d["symbol"] == recommended_pair:
                pattern_name = d.get("chart_pattern", "")
                break
        chart_path = generate_chart_image(df_cache[recommended_pair], recommended_pair, pattern_name)

    return header + result, chart_path


async def run_scan(mode="all", specific_pair=None):
    """Async wrapper"""
    global is_scanning

    if is_scanning:
        return "Already scanning, please wait...", None

    is_scanning = True
    try:
        loop = asyncio.get_event_loop()
        result, chart_path = await loop.run_in_executor(
            executor, blocking_scan, mode, specific_pair
        )
        return result, chart_path
    finally:
        is_scanning = False


async def send_with_chart(chat_id, text, chart_path):
    """Send message with chart image"""
    if chart_path and os.path.exists(chart_path):
        try:
            photo = FSInputFile(chart_path)
            if len(text) <= 1024:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption=text)
            else:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption="Chart — 15m")
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                for chunk in chunks:
                    await bot.send_message(chat_id=chat_id, text=chunk)
                    await asyncio.sleep(0.3)
            os.remove(chart_path)
        except Exception as e:
            logger.error(f"Error sending chart: {e}")
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                await bot.send_message(chat_id=chat_id, text=chunk)
    else:
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            await bot.send_message(chat_id=chat_id, text=chunk)
            await asyncio.sleep(0.3)


# ============================================
# TELEGRAM COMMANDS
# ============================================

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.reply(
        "🤖 SCALP SCANNER BOT\n\n"
        "📊 SCANNING:\n"
        "/scan - Full market scan (top 30)\n"
        "/scan BTC - Deep scan single pair\n"
        "/scan meme - Volatile pairs only\n"
        "/top - Quick top 10 movers\n"
        "/autoscan on/off - Auto-scan every 15 min\n\n"
        "💬 DEEPSEEK CHAT:\n"
        "/ask <question> - Chat with DeepSeek AI\n"
        "/clear - Clear chat history\n\n"
        "⚙️ SETTINGS:\n"
        "/prompt <text> - Customize scan instructions\n"
        "/resetprompt - Reset to default\n"
        "/leverage - Set leverage (default 50x)\n"
        "/status - Bot health check"
    )


@dp.message(Command("scan"))
async def cmd_scan(message: types.Message, command: CommandObject):
    args = command.args if command.args else ""
    args = args.strip().lower()

    if args == "meme":
        await message.reply("Scanning volatile pairs... ~30 seconds")
        result, chart_path = await run_scan(mode="meme")
    elif args:
        await message.reply(f"Scanning {args.upper()}... ~15 seconds")
        result, chart_path = await run_scan(specific_pair=args)
    else:
        await message.reply("Scanning top 30 pairs... ~60 seconds")
        result, chart_path = await run_scan(mode="all")

    await send_with_chart(message.chat.id, result, chart_path)


@dp.message(Command("top"))
async def cmd_top(message: types.Message):
    await message.reply("Fetching top movers...")

    try:
        tickers = exchange.fetch_tickers()
        pairs = []
        skip = ["USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP"]
        for symbol, t in tickers.items():
            if not symbol.endswith("/USDT") or "/USDT:" in symbol:
                continue
            base = symbol.split("/")[0]
            if base in skip:
                continue
            vol = t.get("quoteVolume") or 0
            if vol < 100_000:
                continue
            pairs.append({
                "symbol": base,
                "price": t.get("last", 0),
                "change": t.get("percentage", 0) or 0,
                "volume": vol
            })

        movers = sorted(pairs, key=lambda x: abs(x["change"]), reverse=True)[:10]

        lines = ["TOP 10 MOVERS\n"]
        for i, m in enumerate(movers, 1):
            emoji = "+" if m["change"] > 0 else ""
            lines.append(f"{i}. {m['symbol']}: ${m['price']:.4f} ({emoji}{m['change']:.1f}%)")

        await message.reply("\n".join(lines))
    except Exception as e:
        await message.reply(f"Error: {e}")


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    lines = ["BOT STATUS\n"]

    try:
        exchange.fetch_ticker("BTC/USDT")
        lines.append("Binance: CONNECTED")
    except:
        lines.append("Binance: DISCONNECTED")

    lines.append(f"DeepSeek: {'CONFIGURED' if DEEPSEEK_API_KEY else 'NOT CONFIGURED'}")
    lines.append(f"Leverage: {user_leverage}x")

    if last_scan_time:
        lines.append(f"Last scan: {last_scan_time.strftime('%H:%M UTC')}")
    else:
        lines.append("Last scan: never")

    lines.append(f"Auto-scan: {'ON' if autoscan_enabled else 'OFF'}")
    lines.append(f"Scanning: {'yes' if is_scanning else 'no'}")

    await message.reply("\n".join(lines))


@dp.message(Command("leverage"))
async def cmd_leverage(message: types.Message, command: CommandObject):
    global user_leverage

    args = (command.args or "").strip()

    if not args:
        await message.reply(f"Current leverage: {user_leverage}x\nUse /leverage 50 to change")
        return

    try:
        new_lev = int(args)
        if new_lev < 1 or new_lev > 125:
            await message.reply("Leverage must be between 1 and 125")
            return
        user_leverage = new_lev
        liq_pct = round(1 / new_lev * 100 * 0.95, 2)
        warning = "\nWARNING: Extremely high leverage!" if new_lev > 50 else ""
        await message.reply(f"Leverage set to {new_lev}x\nLiquidation distance: ~{liq_pct}% against you{warning}")
    except ValueError:
        await message.reply("Invalid number. Use /leverage 50")


@dp.message(Command("autoscan"))
async def cmd_autoscan(message: types.Message, command: CommandObject):
    global autoscan_enabled

    args = (command.args or "").strip().lower()

    if args == "on":
        autoscan_enabled = True
        if not scheduler.get_job("autoscan"):
            scheduler.add_job(auto_scan_job, "cron", minute="1,16,31,46", id="autoscan")
        await message.reply("Auto-scan: ON\nScanning at xx:01, xx:16, xx:31, xx:46")

    elif args == "off":
        autoscan_enabled = False
        job = scheduler.get_job("autoscan")
        if job:
            job.remove()
        await message.reply("Auto-scan: OFF")

    else:
        await message.reply(f"Auto-scan: {'ON' if autoscan_enabled else 'OFF'}\nUse /autoscan on or /autoscan off")


# ============================================
# DEEPSEEK CHAT & PROMPT COMMANDS
# ============================================

@dp.message(Command("ask"))
async def cmd_ask(message: types.Message, command: CommandObject):
    """Chat directly with DeepSeek AI"""
    global deepseek_chat_history

    logger.info("ASK command received")

    question = (command.args or "").strip()
    if not question:
        await message.reply(
            "CHAT WITH DEEPSEEK\n\n"
            "Ask anything about crypto, trading, markets:\n"
            "/ask what's a good SL strategy for scalping?\n"
            "/ask explain bull flag pattern\n"
            "/ask is RSI 72 overbought for BTC?\n\n"
            "/clear - Clear chat history"
        )
        return

    await message.reply("Thinking...")

    # Build conversation with history (keep last 10 messages)
    deepseek_chat_history.append({"role": "user", "content": question})
    if len(deepseek_chat_history) > 20:
        deepseek_chat_history = deepseek_chat_history[-20:]

    try:
        def blocking_ask():
            url = "https://api.deepseek.com/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": (
                        "You are an expert crypto trading assistant. "
                        "You specialize in 15-minute scalp trading, technical analysis, "
                        "risk management, and leverage trading. "
                        "Be concise but thorough. Use examples when helpful."
                    )},
                    *deepseek_chat_history
                ],
                "max_tokens": 2000,
                "temperature": 0.5
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            return response

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, blocking_ask)

        if response.status_code != 200:
            await message.reply(f"DeepSeek error: {response.status_code}")
            return

        reply = response.json()["choices"][0]["message"]["content"]
        deepseek_chat_history.append({"role": "assistant", "content": reply})

        # Send in chunks if long
        chunks = [reply[i:i+4000] for i in range(0, len(reply), 4000)]
        for chunk in chunks:
            await message.reply(chunk)
            await asyncio.sleep(0.3)

    except Exception as e:
        logger.error(f"ASK error: {e}")
        await message.reply(f"Error: {e}")


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    """Clear DeepSeek chat history"""
    global deepseek_chat_history
    deepseek_chat_history = []
    await message.reply("💬 Chat history cleared.")


@dp.message(Command("prompt"))
async def cmd_prompt(message: types.Message, command: CommandObject):
    """Modify DeepSeek's trading instructions for scans"""
    global custom_trading_prompt

    args = (command.args or "").strip()

    if not args:
        current = custom_trading_prompt if custom_trading_prompt else "(default - no custom instructions)"
        await message.reply(
            f"📝 CUSTOM SCAN PROMPT\n\n"
            f"Current instructions:\n{current}\n\n"
            f"Examples:\n"
            f"/prompt SL must be tight, max 1% from entry\n"
            f"/prompt only show LONG setups\n"
            f"/prompt focus on BTC ETH SOL only\n"
            f"/prompt risk:reward minimum 1:3\n"
            f"/prompt prefer patterns with >70% confidence\n\n"
            f"/resetprompt - Remove custom instructions"
        )
        return

    custom_trading_prompt = args
    await message.reply(
        f"✅ Custom instructions set!\n\n"
        f"DeepSeek will now follow:\n\"{custom_trading_prompt}\"\n\n"
        f"This applies to all future /scan commands.\n"
        f"Use /resetprompt to remove."
    )


@dp.message(Command("resetprompt"))
async def cmd_resetprompt(message: types.Message):
    """Reset custom prompt to default"""
    global custom_trading_prompt
    custom_trading_prompt = ""
    await message.reply("✅ Custom instructions removed. Using default scan prompt.")


@dp.edited_message()
async def handle_edited(message: types.Message):
    """Handle edited messages as if they were new"""
    text = (message.text or "").strip()
    logger.info(f"Edited message: {text}")
    if text.startswith("/"):
        await catch_all(message)


@dp.message()
async def catch_all(message: types.Message):
    """Catch any unhandled messages"""
    text = (message.text or "").strip()
    logger.info(f"Unhandled message: {text}")

    # Handle /ask manually if Command filter didn't catch it
    if text.lower().startswith("/ask"):
        question = text[4:].strip()
        if question.startswith("@"):
            question = question.split(" ", 1)[1] if " " in question else ""
        question = question.strip()

        if not question:
            await message.reply(
                "CHAT WITH DEEPSEEK\n\n"
                "Usage: /ask <your question>\n"
                "Example: /ask what's a good SL for scalping?"
            )
            return

        await message.reply("Thinking...")

        global deepseek_chat_history
        deepseek_chat_history.append({"role": "user", "content": question})
        if len(deepseek_chat_history) > 20:
            deepseek_chat_history = deepseek_chat_history[-20:]

        try:
            def blocking_ask():
                url = "https://api.deepseek.com/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": (
                            "You are an expert crypto trading assistant. "
                            "You specialize in 15-minute scalp trading, technical analysis, "
                            "risk management, and leverage trading. "
                            "Be concise but thorough. Use examples when helpful."
                        )},
                        *deepseek_chat_history
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.5
                }
                return requests.post(url, headers=headers, json=payload, timeout=60)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(executor, blocking_ask)

            if response.status_code != 200:
                await message.reply(f"DeepSeek error: {response.status_code}")
                return

            reply = response.json()["choices"][0]["message"]["content"]
            deepseek_chat_history.append({"role": "assistant", "content": reply})

            chunks = [reply[i:i+4000] for i in range(0, len(reply), 4000)]
            for chunk in chunks:
                await message.reply(chunk)
                await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"ASK error: {e}")
            await message.reply(f"Error: {e}")


async def auto_scan_job():
    """Auto-scan every 15 min"""
    if not autoscan_enabled or not TELEGRAM_CHAT_ID:
        return

    result, chart_path = await run_scan(mode="all")

    has_setup = any(word in result.upper() for word in ["LONG", "SHORT"])
    if has_setup:
        result = "AUTO-SCAN (15m candle)\n\n" + result
        await send_with_chart(TELEGRAM_CHAT_ID, result, chart_path)


async def main():
    scheduler.start()

    print("=" * 50)
    print("TELEGRAM SCALP SCANNER BOT")
    print("With: Chart Patterns + Leverage + Chart Images")
    print("=" * 50)
    print(f"Leverage: {user_leverage}x")
    print("Waiting for commands...")
    print("=" * 50)

    await dp.start_polling(bot, allowed_updates=["message", "edited_message"])


if __name__ == "__main__":
    asyncio.run(main())
