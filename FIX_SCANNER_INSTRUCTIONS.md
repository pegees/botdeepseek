# FIX INSTRUCTIONS: Make the Scanner Actually Work

> **Problem:** The bot says "no signals above threshold" because it's either not pulling real data from Binance, not calculating indicators properly, or the scoring logic is broken. The scanner needs to be rebuilt from scratch with a simple, working pipeline.
>
> **What we want:** Scan Binance → Pull real candle data → Calculate TA → Package everything into a data dump → Send to DeepSeek → DeepSeek returns a trade setup → Bot posts it to Telegram.
>
> **Philosophy:** DeepSeek is the brain. We don't need a fancy scoring system. We need to GIVE DeepSeek the raw data and let it analyze. The bot is just a data scraper + delivery system.

---

## THE CORE PROBLEM WITH THE CURRENT BOT

The current bot likely does one of these wrong things:

1. **Doesn't actually fetch candle data** — it might just check price and volume from the ticker, which is useless for TA
2. **Calculates indicators wrong** — might use the wrong periods, wrong timeframe, or not enough historical candles
3. **Scoring threshold is too strict** — a score of 65 might be unreachable if the scoring math is off
4. **Doesn't send real data to DeepSeek** — might just send "analyze BTC" instead of actual OHLCV + indicators

**The fix:** Forget the scoring system entirely. Pull raw data, calculate indicators, dump EVERYTHING to DeepSeek, and let DeepSeek decide if there's a setup.

---

## NEW ARCHITECTURE (Simple & Working)

```
[Binance API] → fetch top 30 pairs by volume
                    ↓
         For each pair: fetch 200x 15m candles
                    ↓
         Calculate: RSI, EMA, MACD, Bollinger, ATR, Volume, VWAP
                    ↓
         Package into a structured text summary per pair
                    ↓
         Send ALL pair summaries to DeepSeek in ONE prompt
                    ↓
         DeepSeek picks the best 1-3 setups and returns full trade plan
                    ↓
         Bot formats and posts to Telegram
```

No scoring. No thresholds. DeepSeek sees real numbers and decides.

---

## STEP-BY-STEP BUILD INSTRUCTIONS

### Step 1: Fetch Top Volume Pairs from Binance

Use the Binance REST API (or ccxt) to get all USDT pairs, sorted by 24h quote volume.

```python
import ccxt
import pandas as pd
import pandas_ta as ta

exchange = ccxt.binance({"enableRateLimit": True})
exchange.load_markets()

# Fetch all tickers
tickers = exchange.fetch_tickers()

# Filter: USDT pairs, spot only, volume > $500K
pairs = []
skip = ["USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP"]
for symbol, t in tickers.items():
    if not symbol.endswith("/USDT"):
        continue
    if "/USDT:" in symbol:  # skip futures
        continue
    base = symbol.split("/")[0]
    if base in skip:
        continue
    vol = t.get("quoteVolume") or 0
    if vol < 500_000:
        continue
    pairs.append({"symbol": symbol, "volume": vol, "price": t.get("last", 0), "change": t.get("percentage", 0)})

# Sort by volume, take top 30
pairs.sort(key=lambda x: x["volume"], reverse=True)
top_pairs = pairs[:30]
```

**Critical:** This must return real pairs with real volumes. Print the list to verify. If this returns empty, the API connection is broken.

### Step 2: Fetch 15m Candles & Calculate TA for Each Pair

For each of the top 30 pairs, fetch 200 candles on 15m timeframe and calculate indicators.

```python
def analyze_pair(exchange, symbol):
    """
    Fetch 200 candles of 15m data and calculate all indicators.
    Returns a dict with all the data DeepSeek needs.
    """
    # Fetch candles
    ohlcv = exchange.fetch_ohlcv(symbol, "15m", limit=200)
    if len(ohlcv) < 100:
        return None

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # --- Calculate indicators ---
    df["rsi"] = ta.rsi(df["close"], length=14)

    df["ema9"] = ta.ema(df["close"], length=9)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)

    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df = pd.concat([df, macd_df], axis=1)

    bb_df = ta.bbands(df["close"], length=20, std=2)
    if bb_df is not None:
        df = pd.concat([df, bb_df], axis=1)

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["volume_sma"]

    # --- Extract last candle values ---
    c = df.iloc[-1]  # current (last closed candle)
    p = df.iloc[-2]  # previous candle

    # EMA structure
    ema_stack = "NONE"
    if pd.notna(c.get("ema9")) and pd.notna(c.get("ema21")) and pd.notna(c.get("ema50")):
        if c["ema9"] > c["ema21"] > c["ema50"]:
            ema_stack = "BULLISH (9>21>50)"
        elif c["ema9"] < c["ema21"] < c["ema50"]:
            ema_stack = "BEARISH (9<21<50)"
        else:
            ema_stack = "MIXED"

    # EMA cross detection
    ema_cross = "NONE"
    if pd.notna(c.get("ema9")) and pd.notna(p.get("ema9")):
        if p["ema9"] <= p["ema21"] and c["ema9"] > c["ema21"]:
            ema_cross = "BULLISH CROSS (9 crossed above 21)"
        elif p["ema9"] >= p["ema21"] and c["ema9"] < c["ema21"]:
            ema_cross = "BEARISH CROSS (9 crossed below 21)"

    # MACD
    macd_status = "N/A"
    hist = c.get("MACDh_12_26_9")
    prev_hist = p.get("MACDh_12_26_9")
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
    bbu = c.get("BBU_20_2.0")
    bbl = c.get("BBL_20_2.0")
    if pd.notna(bbu) and pd.notna(bbl) and bbu != bbl:
        pct = (c["close"] - bbl) / (bbu - bbl) * 100
        if pct < 10:
            bb_pos = f"AT LOWER BAND ({pct:.0f}%) — potential bounce"
        elif pct > 90:
            bb_pos = f"AT UPPER BAND ({pct:.0f}%) — potential rejection"
        elif pct < 30:
            bb_pos = f"NEAR LOWER BAND ({pct:.0f}%)"
        elif pct > 70:
            bb_pos = f"NEAR UPPER BAND ({pct:.0f}%)"
        else:
            bb_pos = f"MID RANGE ({pct:.0f}%)"

    # Volume
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
            vol_status = f"VERY LOW ({vr:.1f}x average) — no participation"

    # ATR
    atr_val = c.get("atr")
    atr_pct = (atr_val / c["close"] * 100) if pd.notna(atr_val) else 0

    # Support/Resistance from recent swing highs/lows
    recent = df.tail(50)
    supports = sorted(recent["low"].nsmallest(5).unique())[:3]
    resistances = sorted(recent["high"].nlargest(5).unique(), reverse=True)[:3]

    # Last 5 candles summary (price action context)
    last_5 = []
    for i in range(-5, 0):
        row = df.iloc[i]
        candle_type = "GREEN" if row["close"] > row["open"] else "RED"
        body_pct = abs(row["close"] - row["open"]) / row["open"] * 100
        last_5.append(f"{candle_type} body={body_pct:.2f}% vol_ratio={row['vol_ratio']:.1f}x" if pd.notna(row.get('vol_ratio')) else f"{candle_type} body={body_pct:.2f}%")

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
```

**Critical:** The `analyze_pair` function must return REAL numbers. If RSI is always "N/A", the candle fetch is broken. Add a print statement to debug: `print(f"{symbol}: RSI={result['rsi']}, EMA={result['ema_stack']}")`.

### Step 3: Run the Scanner on All Top Pairs

```python
def scan_all_pairs(exchange, top_pairs):
    """Analyze all top pairs and collect results"""
    results = []
    
    for i, pair_info in enumerate(top_pairs):
        symbol = pair_info["symbol"]
        print(f"  Scanning [{i+1}/{len(top_pairs)}] {symbol}...")
        
        try:
            analysis = analyze_pair(exchange, symbol)
            if analysis:
                analysis["volume_24h"] = pair_info["volume"]
                analysis["change_24h"] = pair_info["change"]
                results.append(analysis)
        except Exception as e:
            print(f"  ⚠️ Error on {symbol}: {e}")
            continue
        
        # Rate limit — Binance allows ~1200 requests/min
        # Each pair = 1 OHLCV request. Sleep 0.5s to be safe.
        import time
        time.sleep(0.5)
    
    return results
```

### Step 4: Build the DeepSeek Prompt

This is the MOST IMPORTANT part. Package ALL the data into a structured prompt and send it to DeepSeek. Do NOT filter or score first — let DeepSeek see everything.

```python
def build_deepseek_prompt(all_pair_data):
    """
    Build a structured prompt with ALL pair data.
    DeepSeek will analyze and pick the best setups.
    """
    
    # Build pair summaries
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
Last 5 candles (newest last): {' | '.join(d['last_5_candles'])}
"""
    
    prompt = f"""You are an expert crypto scalp trader specializing in the 15-minute timeframe. You make decisions based PURELY on data — zero emotion, only probability.

I just scanned the top {len(all_pair_data)} Binance USDT pairs by volume. Here is the REAL technical data for each, calculated from the last 200 candles on the 15m timeframe.

CURRENT SCAN DATA:
{pair_summaries}

YOUR TASK:
Analyze ALL pairs above and find the TOP 1-3 BEST scalp setups right now. For each setup, provide:

1. **PAIR**: Which pair
2. **DIRECTION**: LONG or SHORT
3. **WHY**: Explain exactly which data points create confluence (RSI + EMA + Volume + MACD + Bollinger — what specifically lines up?)
4. **ENTRY**: Exact entry price or zone
5. **STOP LOSS**: Based on the ATR value provided — use 1.5x ATR from entry
6. **TP1**: 1.5R (close 40% here, move SL to breakeven)
7. **TP2**: 2.5R (close 40% here)
8. **TP3**: 4R (close remaining 20%, this is the runner)
9. **CONFIDENCE**: Rate 1-10 based on how many indicators align
10. **RISK**: What could go wrong? What invalidates this setup?

RULES:
- Only pick setups where AT LEAST 3 indicators agree (e.g. RSI oversold + EMA bullish + volume spike)
- Skip pairs with "VERY LOW" volume — no liquidity = bad fills
- Skip pairs with ATR < 0.3% — not enough movement to scalp
- Prefer pairs with fresh EMA crosses or RSI extremes (<30 or >70)
- Prefer pairs where volume is spiking (>1.5x average)
- If you see a setup where RSI is oversold + price at support + volume spike + EMA starting to cross = that's an A+ setup
- If nothing looks great, say "NO CLEAR SETUP — WAIT" instead of forcing a bad trade

FORMAT YOUR RESPONSE AS:
For each setup, use this exact format:

🟢/🔴 [LONG/SHORT] — [PAIR]
Confidence: [X]/10

📊 CONFLUENCE:
- [Indicator 1]: [value] — [why it matters]
- [Indicator 2]: [value] — [why it matters]
- [Indicator 3]: [value] — [why it matters]

💰 TRADE PLAN:
- Entry: [price]
- Stop Loss: [price] ([X]%)
- TP1: [price] (+[X]%) — close 40%, move SL to breakeven
- TP2: [price] (+[X]%) — close 40%
- TP3: [price] (+[X]%) — close 20% runner
- R:R Ratio: 1:[X]

⚠️ INVALIDATION: [what kills this trade]
"""
    return prompt
```

### Step 5: Send to DeepSeek API

```python
def ask_deepseek(prompt):
    """Send the data dump to DeepSeek and get trade setups back"""
    
    url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": DEEPSEEK_MODEL,
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
        "temperature": 0.3  # Low temp = more analytical, less creative
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ DeepSeek API error: {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    return data["choices"][0]["message"]["content"]
```

### Step 6: Send to Telegram

```python
def send_to_telegram(bot_token, channel_id, message):
    """Send the DeepSeek analysis to Telegram"""
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Telegram has a 4096 char limit per message
    # Split if needed
    chunks = []
    if len(message) > 4000:
        # Split by setup (each starts with 🟢 or 🔴)
        parts = message.split("\n🟢")
        parts2 = []
        for p in parts:
            parts2.extend(p.split("\n🔴"))
        
        current_chunk = ""
        for part in parts2:
            if len(current_chunk) + len(part) > 3900:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += "\n" + part
        if current_chunk:
            chunks.append(current_chunk)
    else:
        chunks = [message]
    
    for chunk in chunks:
        payload = {
            "chat_id": channel_id,
            "text": chunk,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload)
            if resp.status_code != 200:
                # Try without markdown if parsing fails
                payload["parse_mode"] = None
                requests.post(url, json=payload)
        except Exception as e:
            print(f"Telegram error: {e}")
```

### Step 7: Main Function — Wire It All Together

```python
def run_scan():
    """MAIN FUNCTION — Run the full pipeline"""
    
    print("=" * 50)
    print("🤖 CRYPTO SCALP SCANNER — Starting...")
    print(f"⏱  Timeframe: 15m")
    print(f"📊 Scanning top {TOP_PAIRS_TO_SCAN} pairs by volume")
    print("=" * 50)
    
    # Step 1: Connect
    exchange = get_exchange()
    
    # Step 2: Get top pairs
    top_pairs = get_top_volume_pairs(exchange, TOP_PAIRS_TO_SCAN)
    if not top_pairs:
        print("❌ No pairs found. Check API connection.")
        return
    
    # Step 3: Analyze each pair
    print("\n📈 Running technical analysis on each pair...\n")
    all_data = scan_all_pairs(exchange, top_pairs)
    print(f"\n✅ Successfully analyzed {len(all_data)} pairs")
    
    if not all_data:
        print("❌ No pairs could be analyzed. Check exchange connection.")
        return
    
    # DEBUGGING: Print a quick summary to verify data is real
    print("\n--- QUICK DATA CHECK (first 5 pairs) ---")
    for d in all_data[:5]:
        print(f"  {d['symbol']}: price={d['price']}, RSI={d['rsi']}, EMA={d['ema_stack']}, Vol={d['volume_status']}")
    print("---\n")
    
    # Step 4: Build prompt and send to DeepSeek
    print("🧠 Sending data to DeepSeek for analysis...")
    prompt = build_deepseek_prompt(all_data)
    
    # DEBUG: Print prompt length to verify data is being sent
    print(f"   Prompt length: {len(prompt)} characters ({len(all_data)} pairs included)")
    
    analysis = ask_deepseek(prompt)
    
    if not analysis:
        print("❌ DeepSeek returned no response. Check API key.")
        return
    
    # Step 5: Print result
    print("\n" + "=" * 50)
    print("🎯 DEEPSEEK ANALYSIS RESULT:")
    print("=" * 50)
    print(analysis)
    
    # Step 6: Send to Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = f"🤖 *SCALP SCAN — {timestamp}*\n⏱ Timeframe: 15m | Pairs scanned: {len(all_data)}\n\n"
        send_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, header + analysis)
        print("\n✅ Sent to Telegram!")
    
    return analysis


# Telegram config (fill in your details)
TELEGRAM_BOT_TOKEN = ""    # from @BotFather
TELEGRAM_CHANNEL_ID = ""   # your channel/group ID

if __name__ == "__main__":
    run_scan()
```

---

## HOW THE BOT SHOULD TRIGGER THIS

The current bot probably has a `/scan` command. Replace whatever it does with a call to `run_scan()`. The bot command handler should look like:

```python
@dp.message(Command("scan"))
async def cmd_scan(message):
    await message.reply("🔍 Scanning top 30 pairs on 15m... This takes ~30 seconds.")
    
    try:
        result = run_scan()  # Call the pipeline above
        if result:
            await message.reply(result[:4000])  # Telegram 4096 char limit
        else:
            await message.reply("❌ Scanner failed. Check logs.")
    except Exception as e:
        await message.reply(f"❌ Error: {str(e)}")
```

Or for a scheduled auto-scan (every 15 minutes on candle close):

```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("cron", minute="1,16,31,46")  # 1 min after each 15m candle close
async def auto_scan():
    result = run_scan()
    if result and "LONG" in result or "SHORT" in result:
        send_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, result)

scheduler.start()
```

---

## DEBUGGING CHECKLIST

If the bot still returns "no signals", check these IN ORDER:

### Check 1: Is Binance returning data?

Add this test at the start of your bot:

```python
# TEST: Does Binance actually return candles?
exchange = ccxt.binance({"enableRateLimit": True})
candles = exchange.fetch_ohlcv("BTC/USDT", "15m", limit=5)
print(f"Got {len(candles)} candles")
print(f"Latest candle: {candles[-1]}")
# Should print something like: [1707500000000, 47250.0, 47300.0, 47200.0, 47280.0, 150.5]
# If this fails → API connection is broken
```

### Check 2: Are indicators being calculated?

```python
# TEST: Do indicators actually calculate?
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "15m", limit=200)
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["rsi"] = ta.rsi(df["close"], length=14)
print(f"RSI values (last 3): {df['rsi'].tail(3).tolist()}")
# Should print real numbers like [45.2, 48.7, 52.1]
# If all NaN → pandas_ta is not installed or data is bad
```

### Check 3: Is DeepSeek receiving the data?

```python
# TEST: Print what you're sending to DeepSeek
prompt = build_deepseek_prompt(all_data)
print(f"Prompt length: {len(prompt)}")
print(f"First 500 chars: {prompt[:500]}")
# Should show actual pair data with real numbers
# If empty or all "N/A" → the analyze step is broken
```

### Check 4: Is DeepSeek responding?

```python
# TEST: Direct DeepSeek call
response = ask_deepseek("Say hello and confirm you're working.")
print(f"DeepSeek says: {response}")
# If None → API key is wrong or endpoint is wrong
```

---

## REQUIRED PYTHON PACKAGES

```bash
pip install ccxt pandas pandas_ta numpy requests aiogram
```

Make sure `pandas_ta` is installed — this is what calculates RSI, EMA, MACD, etc. Without it, all indicators will be N/A and nothing will ever trigger.

```bash
# Verify pandas_ta works
python -c "import pandas_ta; print('pandas_ta OK')"
```

---

## SUMMARY OF WHAT TO CHANGE

| Current (broken) | New (working) |
|---|---|
| Bot has internal scoring (threshold 65) | Remove scoring — DeepSeek decides |
| Probably only checks ticker data | Actually fetches 200 candles per pair |
| Might not calculate real TA | Uses pandas_ta for RSI, EMA, MACD, BB, ATR |
| Sends vague prompt to DeepSeek | Sends full structured data dump with all indicator values |
| Returns "no signals" | Returns 1-3 actionable setups with entry/SL/TP |

The key insight: **Don't build a complex scoring engine. Just scrape real data and let DeepSeek be the brain.** Your bot is a data pipeline, not a decision engine.
