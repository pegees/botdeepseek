#!/usr/bin/env python3
"""
SIMPLE SCALP SCANNER
====================
Fetches REAL data from Binance, calculates TA, sends to DeepSeek.
No fancy scoring - DeepSeek is the brain.

Usage:
    python simple_scanner.py
"""
import os
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

# Load API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "API_KEYS.env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Config
TOP_PAIRS_TO_SCAN = 30
TIMEFRAME = "15m"
CANDLES_TO_FETCH = 200


def get_exchange():
    """Initialize Binance connection"""
    config = {"enableRateLimit": True}
    if BINANCE_API_KEY:
        config["apiKey"] = BINANCE_API_KEY
        config["secret"] = BINANCE_API_SECRET
    return ccxt.binance(config)


def get_top_volume_pairs(exchange, limit=30):
    """Get top pairs by 24h volume"""
    print("Fetching top volume pairs...")

    exchange.load_markets()
    tickers = exchange.fetch_tickers()

    # Filter USDT pairs
    skip_bases = ["USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "EUR", "GBP"]
    pairs = []

    for symbol, t in tickers.items():
        if not symbol.endswith("/USDT"):
            continue
        if "/USDT:" in symbol:  # skip futures symbols in spot
            continue

        base = symbol.split("/")[0]
        if base in skip_bases:
            continue

        vol = t.get("quoteVolume") or 0
        if vol < 500_000:  # Min $500K volume
            continue

        pairs.append({
            "symbol": symbol,
            "volume": vol,
            "price": t.get("last", 0),
            "change": t.get("percentage", 0) or 0
        })

    # Sort by volume
    pairs.sort(key=lambda x: x["volume"], reverse=True)
    top = pairs[:limit]

    print(f"Found {len(top)} pairs with >$500K volume")
    return top


def analyze_pair(exchange, symbol):
    """Fetch candles and calculate all indicators for one pair"""

    # Fetch candles
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES_TO_FETCH)
    if len(ohlcv) < 100:
        return None

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Calculate indicators using 'ta' library
    # RSI
    rsi = RSIIndicator(df["close"], window=14)
    df["rsi"] = rsi.rsi()

    # EMAs
    df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()

    # MACD
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    # Volume ratio
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["volume_sma"]

    # Get last candle values
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
            ema_cross = "BULLISH CROSS (9 crossed above 21)"
        elif p["ema9"] >= p["ema21"] and c["ema9"] < c["ema21"]:
            ema_cross = "BEARISH CROSS (9 crossed below 21)"

    # MACD status
    macd_status = "N/A"
    hist = c.get("macd_hist")
    prev_hist = p.get("macd_hist")
    if pd.notna(hist) and pd.notna(prev_hist):
        if prev_hist < 0 and hist > 0:
            macd_status = "BULLISH CROSS (histogram flipped positive)"
        elif prev_hist > 0 and hist < 0:
            macd_status = "BEARISH CROSS (histogram flipped negative)"
        elif hist > 0 and hist > prev_hist:
            macd_status = "BULLISH MOMENTUM (histogram growing)"
        elif hist < 0 and hist < prev_hist:
            macd_status = "BEARISH MOMENTUM (histogram falling)"
        elif hist > 0:
            macd_status = "BULLISH (above zero)"
        else:
            macd_status = "BEARISH (below zero)"

    # Bollinger position
    bb_pos = "N/A"
    bbu = c.get("bb_upper")
    bbl = c.get("bb_lower")
    if pd.notna(bbu) and pd.notna(bbl) and bbu != bbl:
        pct = (c["close"] - bbl) / (bbu - bbl) * 100
        if pct < 10:
            bb_pos = f"AT LOWER BAND ({pct:.0f}%) - potential bounce"
        elif pct > 90:
            bb_pos = f"AT UPPER BAND ({pct:.0f}%) - potential rejection"
        elif pct < 30:
            bb_pos = f"NEAR LOWER BAND ({pct:.0f}%)"
        elif pct > 70:
            bb_pos = f"NEAR UPPER BAND ({pct:.0f}%)"
        else:
            bb_pos = f"MID RANGE ({pct:.0f}%)"

    # Volume status
    vol_status = "NORMAL"
    vr = c.get("vol_ratio")
    if pd.notna(vr):
        if vr > 3:
            vol_status = f"EXTREME SPIKE ({vr:.1f}x average)"
        elif vr > 2:
            vol_status = f"HIGH SPIKE ({vr:.1f}x average)"
        elif vr > 1.5:
            vol_status = f"ABOVE AVERAGE ({vr:.1f}x)"
        elif vr < 0.5:
            vol_status = f"VERY LOW ({vr:.1f}x average) - no participation"

    # ATR
    atr_val = c.get("atr")
    atr_pct = (atr_val / c["close"] * 100) if pd.notna(atr_val) else 0

    # Support/Resistance from recent swings
    recent = df.tail(50)
    supports = sorted(recent["low"].nsmallest(5).unique())[:3]
    resistances = sorted(recent["high"].nlargest(5).unique(), reverse=True)[:3]

    # Last 5 candles
    last_5 = []
    for i in range(-5, 0):
        row = df.iloc[i]
        candle_type = "GREEN" if row["close"] > row["open"] else "RED"
        body_pct = abs(row["close"] - row["open"]) / row["open"] * 100
        vr_str = f"vol={row['vol_ratio']:.1f}x" if pd.notna(row.get('vol_ratio')) else ""
        last_5.append(f"{candle_type} {body_pct:.2f}% {vr_str}")

    return {
        "symbol": symbol,
        "price": round(c["close"], 8),
        "rsi": round(c["rsi"], 1) if pd.notna(c.get("rsi")) else "N/A",
        "ema_stack": ema_stack,
        "ema_cross": ema_cross,
        "ema9": round(c["ema9"], 8) if pd.notna(c.get("ema9")) else "N/A",
        "ema21": round(c["ema21"], 8) if pd.notna(c.get("ema21")) else "N/A",
        "ema50": round(c["ema50"], 8) if pd.notna(c.get("ema50")) else "N/A",
        "macd_status": macd_status,
        "macd_histogram": round(hist, 8) if pd.notna(hist) else "N/A",
        "bollinger_position": bb_pos,
        "volume_status": vol_status,
        "volume_ratio": round(vr, 1) if pd.notna(vr) else "N/A",
        "atr": round(atr_val, 8) if pd.notna(atr_val) else "N/A",
        "atr_pct": round(atr_pct, 2),
        "supports": [round(s, 8) for s in supports],
        "resistances": [round(r, 8) for r in resistances],
        "last_5_candles": last_5
    }


def scan_all_pairs(exchange, top_pairs):
    """Analyze all top pairs"""
    results = []

    for i, pair_info in enumerate(top_pairs):
        symbol = pair_info["symbol"]
        print(f"  [{i+1}/{len(top_pairs)}] Scanning {symbol}...")

        try:
            analysis = analyze_pair(exchange, symbol)
            if analysis:
                analysis["volume_24h"] = pair_info["volume"]
                analysis["change_24h"] = pair_info["change"]
                results.append(analysis)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        time.sleep(0.3)  # Rate limit

    return results


def build_deepseek_prompt(all_pair_data):
    """Build the mega prompt with ALL pair data for DeepSeek"""

    pair_summaries = ""
    for d in all_pair_data:
        pair_summaries += f"""
--- {d['symbol']} ---
Price: {d['price']}
24h Change: {d['change_24h']:.1f}%
24h Volume: ${d['volume_24h']:,.0f}
RSI(14): {d['rsi']}
EMA Structure: {d['ema_stack']}
EMA Cross: {d['ema_cross']}
EMA 9: {d['ema9']} | EMA 21: {d['ema21']} | EMA 50: {d['ema50']}
MACD: {d['macd_status']} (histogram: {d['macd_histogram']})
Bollinger: {d['bollinger_position']}
Volume: {d['volume_status']} (ratio: {d['volume_ratio']}x)
ATR(14): {d['atr']} ({d['atr_pct']}% of price)
Support levels: {d['supports']}
Resistance levels: {d['resistances']}
Last 5 candles: {' | '.join(d['last_5_candles'])}
"""

    prompt = f"""You are an expert crypto scalp trader specializing in the 15-minute timeframe. You make decisions based PURELY on data - zero emotion, only probability.

I just scanned the top {len(all_pair_data)} Binance USDT pairs by volume. Here is the REAL technical data for each, calculated from the last 200 candles on the 15m timeframe.

CURRENT SCAN DATA:
{pair_summaries}

YOUR TASK:
Analyze ALL pairs above and find the TOP 1-3 BEST scalp setups right now. For each setup, provide:

1. **PAIR**: Which pair
2. **DIRECTION**: LONG or SHORT
3. **WHY**: Explain exactly which data points create confluence (RSI + EMA + Volume + MACD + Bollinger - what specifically lines up?)
4. **ENTRY**: Exact entry price or zone
5. **STOP LOSS**: Based on the ATR value provided - use 1.5x ATR from entry
6. **TP1**: 1.5R (close 40% here, move SL to breakeven)
7. **TP2**: 2.5R (close 40% here)
8. **TP3**: 4R (close remaining 20%, this is the runner)
9. **CONFIDENCE**: Rate 1-10 based on how many indicators align
10. **RISK**: What could go wrong? What invalidates this setup?

RULES:
- Only pick setups where AT LEAST 3 indicators agree
- Skip pairs with "VERY LOW" volume - no liquidity = bad fills
- Skip pairs with ATR < 0.3% - not enough movement to scalp
- Prefer pairs with fresh EMA crosses or RSI extremes (<30 or >70)
- Prefer pairs where volume is spiking (>1.5x average)
- If nothing looks great, say "NO CLEAR SETUP - WAIT" instead of forcing a bad trade

FORMAT YOUR RESPONSE AS:

For each setup:

[LONG/SHORT] - [PAIR]
Confidence: [X]/10

CONFLUENCE:
- [Indicator 1]: [value] - [why it matters]
- [Indicator 2]: [value] - [why it matters]
- [Indicator 3]: [value] - [why it matters]

TRADE PLAN:
- Entry: [price]
- Stop Loss: [price] ([X]%)
- TP1: [price] (+[X]%) - close 40%, move SL to BE
- TP2: [price] (+[X]%) - close 40%
- TP3: [price] (+[X]%) - close 20% runner
- R:R Ratio: 1:[X]

INVALIDATION: [what kills this trade]
"""
    return prompt


def ask_deepseek(prompt):
    """Send data to DeepSeek and get analysis"""

    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional cryptocurrency scalp trader. You analyze raw technical data and find high-probability 15-minute timeframe setups. You are data-driven and never emotional. You only recommend trades with strong confluence of at least 3 aligning indicators. If nothing looks good, you say wait."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"DeepSeek API error: {response.status_code}")
        print(response.text)
        return None

    data = response.json()
    return data["choices"][0]["message"]["content"]


def send_to_telegram(message):
    """Send result to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured, skipping...")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Split long messages
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]

    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"Telegram error: {resp.text}")
        except Exception as e:
            print(f"Telegram error: {e}")


def run_scan():
    """MAIN FUNCTION - Run the full pipeline"""

    print("=" * 50)
    print("CRYPTO SCALP SCANNER - Starting...")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Scanning top {TOP_PAIRS_TO_SCAN} pairs by volume")
    print("=" * 50)

    # Step 1: Connect
    exchange = get_exchange()

    # Step 2: Get top pairs
    top_pairs = get_top_volume_pairs(exchange, TOP_PAIRS_TO_SCAN)
    if not top_pairs:
        print("No pairs found. Check API connection.")
        return None

    # Step 3: Analyze each pair
    print(f"\nRunning technical analysis on {len(top_pairs)} pairs...\n")
    all_data = scan_all_pairs(exchange, top_pairs)
    print(f"\nAnalyzed {len(all_data)} pairs successfully")

    if not all_data:
        print("No pairs could be analyzed.")
        return None

    # Debug: Quick check
    print("\n--- DATA CHECK (first 3 pairs) ---")
    for d in all_data[:3]:
        print(f"  {d['symbol']}: RSI={d['rsi']}, EMA={d['ema_stack']}, Vol={d['volume_status']}")
    print("---\n")

    # Step 4: Send to DeepSeek
    print("Sending data to DeepSeek for analysis...")
    prompt = build_deepseek_prompt(all_data)
    print(f"Prompt size: {len(prompt)} characters")

    analysis = ask_deepseek(prompt)

    if not analysis:
        print("DeepSeek returned no response.")
        return None

    # Step 5: Print result
    print("\n" + "=" * 50)
    print("DEEPSEEK ANALYSIS:")
    print("=" * 50)
    print(analysis)

    # Step 6: Send to Telegram
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"SCALP SCAN - {timestamp}\nTimeframe: {TIMEFRAME} | Pairs: {len(all_data)}\n\n"
    send_to_telegram(header + analysis)
    print("\nSent to Telegram!")

    return analysis


if __name__ == "__main__":
    run_scan()
